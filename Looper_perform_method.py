def perform(self, phrase: Phrase, gestures: Optional[Phrase], tempo=None, wait_for_measure_end=False, repeats: int = 1):
        phrase = self.filter_phrase(phrase, min_note_dist_ms=self.min_note_dist_ms,
                                    max_notes_per_onset=self.max_notes_per_onset)
        notes, onsets = phrase.get()
        m = 1
        if self.tempo and tempo:
            m = self.tempo / tempo


        if gestures is not None:
            self.perform_gestures(gestures=gestures, tempo=tempo, wait_for_measure_end=wait_for_measure_end)

        for loop_idx in range(repeats):
            i = 0
            while i < len(notes):
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
