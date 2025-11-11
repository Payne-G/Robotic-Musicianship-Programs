"""
Author: Raghavasimhan Sankaranarayanan
Date: 04/08/2022
"""
import os.path
from copy import copy, deepcopy

from enum import IntEnum
import pretty_midi
from pretty_midi import Note
import pyaudio
from pythonosc import udp_client
from rtmidi.midiconstants import NOTE_OFF, NOTE_ON
from audioToMidi import AudioMidiConverter
from audioDevice import AudioDevice
from tempoTracker import TempoTracker
from gestureController import GestureController
import numpy as np
from threading import Thread, Lock, Event
import time
import threading
from queue import Queue
from typing import Optional


class Instruments:
    def __init__(self, instruments: list[str]):
        self.instruments = instruments
        self.idx = 0
        self.keyboard = "Keys"
        self.violin = "Violin"

    def __len__(self):
        return len(self.instruments)

    def __next__(self):
        self.idx = (self.idx + 1) % len(self.instruments)
        return self.idx

    def current(self):
        return self.instruments[self.idx]


class Phrase:
    def __init__(self, notes=None, onsets=None, tempo=None, name=None):
        self.name = name
        self.notes = notes if notes is not None else []
        self.onsets = onsets if onsets is not None else []
        self.tempo = tempo
        self.is_korvai = name == "korvai"
        self.is_intro = name == "intro"

    def get(self):
        return self.notes, self.onsets

    def get_raw_notes(self):
        notes = []
        for note in self.notes:
            notes.append(note.pitch)
        return np.array(notes)

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, item):
        if len(self.notes) > item:
            return self.notes[item], self.onsets[item]
        return None, None

    def __setitem__(self, key, value: tuple):
        if len(self.notes) > key:
            self.notes[key] = value[0]
            self.onsets[key] = value[1]

    def append(self, note, onset):
        self.notes.append(note)
        self.onsets.append(onset)

    def __str__(self):
        ret = ""
        for i in range(len(self.notes)):
            ret = ret + f"{self.notes[i]}, Onset: {self.onsets[i]}\n"
        return ret


class Performer(GestureController):
    def __init__(self, osc_address: str, osc_port: int, gesture_note_mapping: dict[str, int], osc_arm_route: str = "/arm",
                 osc_head_route: str = "/head", tempo=None, ticks=None, min_note_dist_ms=50,
                 max_notes_per_onset=4):
        self.client = udp_client.SimpleUDPClient(osc_address, osc_port)
        super().__init__(self.client, gesture_note_mapping, osc_head_route)
        self.tempo = tempo
        self.osc_arm_route = osc_arm_route
        self.ticks = ticks
        self.min_note_dist_ms = min_note_dist_ms
        self.max_notes_per_onset = max_notes_per_onset
        self.note_on_thread = Thread()
        self.note_off_thread = Thread()
        self.lock = Lock()
        self.stop_event = threading.Event()
        self.timer = None

        self.is_performing = False
        self.perform_lock = Lock()

    def perform_gestures(self, gestures: Phrase, tempo=None, wait_for_measure_end=False):
        self.note_on_thread = Thread(target=self.handle_note_ons, args=(gestures.notes, tempo))
        self.note_off_thread = Thread(target=self.handle_note_offs, args=(gestures.notes, tempo))
        if self.timer and self.timer.is_alive():
            self.timer.join()
        self.stop_event.set()
        self.timer = threading.Timer(0.5, self.delay_start_thread)
        self.timer.start()

    def delay_start_thread(self):
        if self.note_on_thread.is_alive():
            self.note_on_thread.join()
        if self.note_off_thread.is_alive():
            self.note_off_thread.join()
        self.stop_event.clear()
        self.note_on_thread.start()
        self.note_off_thread.start()

    def handle_note_ons(self, notes: list[Note], tempo: int):
        m = 1
        if tempo and self.tempo:
            m = self.tempo / tempo

        prev_start = 0
        now = time.time()
        for note in notes:
            if self.stop_event.is_set():
                return
            dly = max(0, ((note.start - prev_start) * m) - (time.time() - now))
            self.stop_event.wait(dly)
            # time.sleep(dly)
            now = time.time()
            self.lock.acquire()
            self.send_gesture(note.pitch, note.velocity)
            self.lock.release()
            prev_start = note.start

    def handle_note_offs(self, notes: list[Note], tempo: int):
        m = 1
        if tempo and self.tempo:
            m = self.tempo / tempo

        prev_end = 0
        now = time.time()
        for note in notes:
            if self.stop_event.is_set():
                return
            dly = max(0, ((note.end - prev_end) * m) - (time.time() - now))
            self.stop_event.wait(dly)
            now = time.time()
            self.lock.acquire()
            self.send_gesture(note.pitch, 0)
            self.lock.release()
            prev_end = note.end

    def perform(self, phrase: Phrase, gestures: Optional[Phrase], tempo=None, wait_for_measure_end=False, repeats: int = 1, temperature: float = 0):
        with self.perform_lock:
            if self.is_performing:
                print("Performer is already active, skipping new performance.")
                return
            self.is_performing = True

        
        try:
            phrase = self.filter_phrase(phrase, min_note_dist_ms=self.min_note_dist_ms,
                                        max_notes_per_onset=self.max_notes_per_onset)
            
            m = 1
            if self.tempo and tempo:
                m = self.tempo / tempo


            if gestures is not None:
                self.perform_gestures(gestures=gestures, tempo=tempo, wait_for_measure_end=wait_for_measure_end)

            for loop_idx in range(repeats):
                i = 0
                while i < len(notes):
                    phrase_copy = deepcopy(phrase)
                    phrase_copy = QnADemo.process_midi_phrase(phrase_copy, temperature)

                    notes, onsets = phrase_copy.get()

                    poly_notes = []
                    poly_onsets = []
                    while i < len(onsets):
                        if len(poly_notes) > 0 and poly_onsets[-1] == onsets[i]:
                            poly_notes.append(notes[i])
                            poly_onsets.append(onsets[i])
                            i += 1
                        elif len(poly_notes) == 0 and i < len(notes) - 1 and onsets[i] == onsets[i + 1]:
                            poly_notes.append(notes[i])
                            poly_notes.append(notes[i + 1])
                            poly_onsets.append(onsets[i])
                            poly_onsets.append(onsets[i + 1])
                            i += 2
                        else:
                            break

                    duration = 0
                    if len(poly_notes) > 0:
                        if i < len(notes):
                            duration = notes[i].start - poly_notes[0].start
                        for j in range(len(poly_notes)):
                            note_on = [int(poly_notes[j].pitch), int(poly_notes[j].velocity)]
                            self.client.send_message(self.osc_arm_route, note_on)
                    else:
                        if i < len(notes) - 1:
                            duration = notes[i + 1].start - notes[i].start
                        note_on = [int(notes[i].pitch), int(notes[i].velocity)]
                        self.client.send_message(self.osc_arm_route, note_on)
                        i += 1

                    time.sleep(duration * m)

            if wait_for_measure_end and tempo and self.ticks:
                self.wait_for_measure_end(onsets, tempo)

            # if gestures is not None:
            #     self.note_on_thread.join(0.1)
            #     self.note_off_thread.join(0.1)
        finally:
            with self.perform_lock:
                self.is_performing = False

    def wait_for_measure_end(self, onsets, tempo):
        # Assume 4/4
        bar_tick = self.ticks * 4
        # measure_tick = bar_tick * 4
        while bar_tick < onsets[-1]:
            bar_tick += bar_tick
        remaining_ticks = bar_tick - onsets[-1]
        print(remaining_ticks, bar_tick, onsets[-1])
        if remaining_ticks > 0:
            time.sleep(remaining_ticks * 60 / (tempo * self.ticks))

    @staticmethod
    def filter_phrase(phrase: Phrase, min_note_dist_ms: float = 50, max_notes_per_onset: int = 4):
        temp = Phrase()
        notes, onsets = phrase.get()
        same_onset_count = 0

        min_note_dist = min_note_dist_ms / 1000
        temp.append(notes[0], onsets[0])
        for i in range(1, len(phrase)):
            d_time = abs(notes[i].start - temp[-1][0].start)
            if d_time < 1e-2:
                if same_onset_count >= max_notes_per_onset - 1:
                    continue
                same_onset_count += 1
            elif min_note_dist > abs(notes[i].start - temp[-1][0].start):
                continue
            else:
                same_onset_count = 0
            temp.append(notes[i], onsets[i])

        return temp


class Demo:
    def __init__(self):
        pass


class QnADemo(Demo):
    def __init__(self, performer: Performer, raga_map, sr=16000,
                 instruments=("Violin", "Keys"), frame_size=2048, activation_threshold=0.02, n_wait=16,
                 input_dev_name='Line 6 HX Stomp', outlier_filter_coeff=2, timeout_sec=2):
        super().__init__()
        self.active = False
        self.activation_threshold = activation_threshold
        self.n_wait = n_wait
        self.wait_count = 0
        self.playing = False
        self.phrase = []
        self.midi_notes = []
        self.midi_onsets = []

        self.process_thread = Thread()
        self.event = Event()
        self.lock = Lock()

        try:
            self.audioDevice = AudioDevice(self.callback_fn, rate=sr, frame_size=frame_size,
                                           input_dev_name=input_dev_name,
                                           channels=4)
        except AssertionError:
            print(f"{input_dev_name} not found. Disabling violin input for QnA Demo")
            self.audioDevice = None

        self.audio2midi = AudioMidiConverter(raga_map=raga_map, sr=sr, frame_size=frame_size,
                                             outlier_coeff=outlier_filter_coeff)
        if self.audioDevice:
            self.audioDevice.start()

        self.instruments = Instruments(instruments) if self.audioDevice else Instruments(["Keys"])
        self.timeout = timeout_sec
        self.last_time = time.time()
        self.performer = performer

    def reset_var(self):
        self.wait_count = 0
        self.playing = False
        self.phrase = []
        self.last_time = time.time()

    def handle_midi(self, msg, dt):
        if self.instruments.current() != self.instruments.keyboard:
            print(f"Its {self.instruments.current()}'s turn")
            return

        if msg[0] == NOTE_ON:
            self.last_time = time.time()
            note = pretty_midi.Note(msg[2], msg[1], self.last_time, self.last_time + 0.1)
            self.midi_notes.append(note)
            self.midi_onsets.append(self.last_time)

    def callback_fn(self, in_data: bytes, frame_count: int, time_info: dict[str, float], status: int) -> tuple[
        bytes, int]:
        if not self.active:
            self.reset_var()
            return in_data, pyaudio.paContinue

        y = np.frombuffer(in_data, dtype=np.int16)
        y = y[::2][1::2]  # Get all the even indices then get all odd indices for ch-3 of HX Stomp
        y = self.int16_to_float(y)
        activation = np.abs(y).mean()
        if activation > self.activation_threshold:
            print(activation)
            if self.instruments.current() != self.instruments.violin:
                print(f"Its {self.instruments.current()}'s turn")
                self.reset_var()
                return in_data, pyaudio.paContinue
            self.playing = True
            self.wait_count = 0
            self.lock.acquire()
            self.phrase.append(y)
            self.lock.release()
        else:
            if self.wait_count > self.n_wait:
                self.playing = False
                self.wait_count = 0
            else:
                self.lock.acquire()
                if self.playing:
                    self.phrase.append(y)
                self.lock.release()
                self.wait_count += 1
        return in_data, pyaudio.paContinue

    def reset(self):
        self.stop()
        if self.audioDevice:
            self.audioDevice.reset()

    @staticmethod
    def int16_to_float(x):
        return x / (1 << 15)

    # @staticmethod
    # def to_float(x):
    #     if x.dtype == 'float32':
    #         return x
    #     elif x.dtype == 'uint8':
    #         return (x / 128.) - 1
    #     else:
    #         bits = x.dtype.itemsize * 8
    #         return x / (2 ** (bits - 1))

    def start(self):
        self.reset_var()
        if self.process_thread.is_alive():
            self.process_thread.join()
        self.lock.acquire()
        self.active = True
        self.lock.release()
        self.process_thread = Thread(target=self._process)
        self.process_thread.start()
        self.event.clear()
        self.check_timeout()

    def _process(self):
        while True:
            time.sleep(0.1)
            self.lock.acquire()
            if not self.active:
                self.lock.release()
                return

            if not (self.playing or len(self.phrase) == 0):
                self.lock.release()
                break
            self.lock.release()

        self.lock.acquire()
        phrase = np.hstack(self.phrase)
        self.phrase = []
        self.lock.release()

        if len(phrase) > 0:
            notes, onsets = self.audio2midi.convert(phrase, return_onsets=True)
            print("notes:", notes)  # Send to shimon
            print("onsets:", onsets)
            phrase = Phrase(notes, onsets)
            if self.performer.is_performing:
                print("Skipping phrase — performer is busy.")
                return

            self.perform(phrase)

        self._process()

    def stop(self):
        self.lock.acquire()
        self.active = False
        self.lock.release()
        if self.process_thread.is_alive():
            self.process_thread.join()
        if self.audioDevice:
            self.audioDevice.stop()
        self.event.set()

    def perform(self, phrase):
        self.performer.send_gesture(gesture="look", velocity=3)  # Look straight
        self.performer.send_gesture(gesture="headcircle", velocity=80)
        # threading.Timer(0.5, self.gesture_controller.send, kwargs={"gesture": "headcircle", "velocity": 80}).start()
        # time.sleep(0.5)     # Shimon hardware wait simulation
        self.performer.perform(phrase=phrase, gestures=None, loops = 2, temperature = 0)
        self.performer.send_gesture(gesture="headcircle", velocity=0)
        self.performer.send_gesture(gesture="look",
                                    velocity=next(self.instruments) + 1)  # Look at the respective artist

    @staticmethod
    def process_midi_phrase(phrase, temperature: float = 1.0):
        temperature = max(min(temperature, 1), 0)
        n_notes_to_change = np.random.randint(0, int((len(phrase)) * temperature), 1)
        w = np.hanning(len(phrase)) + 1e-6  # to avoid ValueError: Fewer non-zero entries in p than size
        p = w / np.sum(w)
        indices = np.random.choice(np.arange(len(phrase)), n_notes_to_change, replace=False, p=p)
        options = np.unique(phrase.get_raw_notes())
        for i in indices:
            phrase.notes[i].pitch = np.random.choice(options, 1)[0]
        return phrase

    def check_timeout(self):
        if time.time() - self.last_time > self.timeout and len(self.midi_notes) > 0:
            midi_notes = copy(self.midi_notes)
            midi_onsets = copy(self.midi_onsets)
            self.midi_notes = []
            self.midi_onsets = []
            t = midi_notes[0].start
            for i in range(len(midi_notes)):
                midi_notes[i].start -= t
                midi_notes[i].end -= t
                midi_onsets[i] -= t

            phrase = Phrase(midi_notes, midi_onsets)
            phrase = self.process_midi_phrase(phrase)
            if self.performer.is_performing:
                print("Performer busy, delaying new phrase.")
                threading.Timer(1, self.check_timeout).start()
                return

            self.perform(phrase)

        if not self.event.is_set():
            threading.Timer(1, self.check_timeout).start()


class BeatDetectionDemo(Demo):
    def __init__(self, performer: Performer, tempo_range: tuple = (60, 120), smoothing=4, n_beats_to_track=16,
                 timeout_sec=5, timeout_callback=None, user_data=None, default_tempo: int = 80):
        super().__init__()
        self.performer = performer
        self.timeout_callback = timeout_callback
        self.user_data = user_data
        self.tempo_tracker = TempoTracker(smoothing=smoothing, n_beats_to_track=n_beats_to_track,
                                          tempo_range=tempo_range, default_tempo=default_tempo,
                                          timeout_sec=timeout_sec, timeout_callback=self.timeout_handle)
        self._event = threading.Event()
        self._first_time = True
        self._last_time = time.time()
        self._beat_interval = -1

    def start(self):
        self._first_time = True
        self.performer.send_gesture("look", 8)  # look at the keyboard artist
        self.tempo_tracker.start()
        self._event.clear()

    def stop(self):
        self.tempo_tracker.stop()
        self._event.set()
        self._first_time = True

    def reset(self):
        self.stop()

    def handle_midi(self, msg, dt):
        self.update_tempo(msg, dt)

    def update_tempo(self, msg, dt):
        if msg[0] == NOTE_ON:
            tempo = self.tempo_tracker.track_tempo(msg, dt)
            if tempo:
                print(tempo)
                self.set_beat_interval(tempo)
                if self._first_time:
                    self.gesture_ctl()
                    self._first_time = False

    def set_beat_interval(self, tempo: float):
        self._beat_interval = 60 / tempo

    def get_tempo(self):
        return self.tempo_tracker.tempo

    def timeout_handle(self):
        self.timeout_callback(self.user_data)

    def gesture_ctl(self):
        self.performer.send_gesture("beatOnce", 80)
        if not self._event.is_set():
            threading.Timer(self._beat_interval, self.gesture_ctl).start()


class SongDemo(Demo):
    def __init__(self, performer: Performer, midi_files: list[list[str]], gesture_midi_files: list[list[str]],
                 start_note_for_phrase_mapping: int = 36, complete_callback=None, user_data=None):
        super().__init__()
        self.performer = performer
        self.phrase_note_map = start_note_for_phrase_mapping
        self.user_data = user_data
        self.phrase_idx = 0
        self.variation_idx = 0
        self.phrases = self._parse_midi(midi_files)
        self.g_phrases = self._parse_midi(gesture_midi_files)
        self.file_tempo = self.phrases[self.phrase_idx][self.variation_idx].tempo
        self.next_phrase = self.phrases[self.phrase_idx][self.variation_idx]  # intro phrase
        self.next_g_phrase = self.g_phrases[self.phrase_idx][self.variation_idx]  # intro gesture
        self.ticks = 480
        self.tempo = self.file_tempo
        self.playing = False
        self.thread = Thread()
        self.lock = Lock()
        self.callback_queue = Queue(1)
        self.callback_queue.put(complete_callback)

    def __del__(self):
        self.reset()

    def set_tempo(self, tempo):
        self.tempo = tempo

    def start(self):
        self.playing = True
        self.performer.send_gesture("look", 8)  # look at the keyboard artist
        self.thread = Thread(target=self.perform, args=(self.next_phrase, self.next_g_phrase))
        self.thread.start()

    def stop(self):
        self.lock.acquire()
        self.playing = False
        self.lock.release()

    def handle_midi(self, msg, dt):
        if msg[0] == NOTE_ON:
            # print(msg)
            idx = msg[1] - self.phrase_note_map
            if 0 <= idx < len(self.phrases):
                self.phrase_idx = idx
                self.set_phrase(reset_variation=True)

    def set_phrase(self, reset_variation: bool = False):
        idx = self.phrase_idx
        if len(self.phrases) > idx >= 0:
            self.variation_idx = 0 if reset_variation else (self.variation_idx + 1) % len(self.phrases[idx])
            self.next_phrase = self.phrases[idx][self.variation_idx]
            self.next_g_phrase = self.g_phrases[idx][self.variation_idx]
            print(self.next_phrase.name)

    def perform(self, phrase: Optional[Phrase], gestures: Optional[Phrase]):
        if phrase.is_korvai:
            self.next_phrase = None
            self.next_g_phrase = None

        if phrase.is_intro and len(self.phrases) > 1:
            self.phrase_idx = 1

        self.performer.perform(phrase, gestures, self.tempo, wait_for_measure_end=True)

        if self.next_phrase and self.next_g_phrase:
            self.set_phrase()  # Calling this here will cycle variation
            self.perform(phrase=self.next_phrase, gestures=self.next_g_phrase)

    def wait(self):
        if self.thread.is_alive():
            self.thread.join()

    def _parse_midi(self, midi_files):
        if not midi_files:
            return None

        # Func to use as key for the sort method
        def note_sort(_note):
            return _note.start

        phrases = []

        for variations in midi_files:
            temp = []
            for midi_file in variations:
                name = os.path.splitext(os.path.split(midi_file)[-1])[0]
                midi_data = pretty_midi.PrettyMIDI(midi_file)
                self.ticks = midi_data.resolution
                notes = sorted(midi_data.instruments[0].notes, key=note_sort)
                onsets = []
                for note in notes:
                    onsets.append(midi_data.time_to_tick(note.start))
                # print(name)
                # print(notes)
                # print(onsets)
                # print()
                temp.append(Phrase(notes, onsets, round(midi_data.get_tempo_changes()[1][0], 3), name))
            phrases.append(temp)
        return phrases

    def reset(self):
        self.stop()
        self.wait()
