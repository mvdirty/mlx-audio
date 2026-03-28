import unittest
from unittest import mock

import numpy as np
import sounddevice as sd

from mlx_audio.tts.audio_player import AudioPlayer


class TestAudioPlayer(unittest.TestCase):
    def test_callback_keeps_stream_alive_on_underrun_before_finish(self):
        player = AudioPlayer(sample_rate=24_000, buffer_size=4)
        player.playing = True
        player.audio_buffer.append(np.array([0.1, 0.2], dtype=np.float32))

        outdata = np.empty((4, 1), dtype=np.float32)
        player.callback(outdata, 4, None, None)

        np.testing.assert_allclose(outdata[:, 0], [0.1, 0.2, 0.0, 0.0])
        self.assertFalse(player.drain_event.is_set())
        self.assertTrue(player.playing)
        self.assertEqual(len(player.audio_buffer), 0)

    def test_callback_stops_only_after_empty_callback_once_input_is_closed(self):
        player = AudioPlayer(sample_rate=24_000, buffer_size=4)
        player.playing = True
        player.input_closed = True
        player.audio_buffer.append(np.array([0.1, 0.2], dtype=np.float32))

        outdata = np.empty((4, 1), dtype=np.float32)
        player.callback(outdata, 4, None, None)

        np.testing.assert_allclose(outdata[:, 0], [0.1, 0.2, 0.0, 0.0])
        self.assertFalse(player.drain_event.is_set())
        self.assertTrue(player.playing)
        self.assertEqual(len(player.audio_buffer), 0)

        outdata = np.empty((4, 1), dtype=np.float32)
        with self.assertRaises(sd.CallbackStop):
            player.callback(outdata, 4, None, None)

        np.testing.assert_allclose(outdata[:, 0], [0.0, 0.0, 0.0, 0.0])
        self.assertTrue(player.drain_event.is_set())
        self.assertFalse(player.playing)

    def test_finish_starts_stream_for_small_remaining_buffer(self):
        player = AudioPlayer(sample_rate=24_000, buffer_size=2048)
        player.arrival_rate = 24_000

        with mock.patch.object(player, "start_stream") as mock_start_stream:
            player.queue_audio(np.ones(10, dtype=np.float32))
            mock_start_stream.assert_not_called()

            player.finish()

        mock_start_stream.assert_called_once_with()
        self.assertTrue(player.input_closed)
        self.assertFalse(player.drain_event.is_set())

    def test_finish_without_buffer_sets_drain_event(self):
        player = AudioPlayer(sample_rate=24_000, buffer_size=2048)

        player.finish()

        self.assertTrue(player.input_closed)
        self.assertTrue(player.drain_event.is_set())
        self.assertFalse(player.playing)

    def test_stop_does_not_wait_for_drain_when_input_is_not_closed(self):
        player = AudioPlayer(sample_rate=24_000, buffer_size=2048)
        player.playing = True

        with (
            mock.patch.object(player, "wait_for_drain") as mock_wait_for_drain,
            mock.patch("mlx_audio.tts.audio_player.sd.sleep") as mock_sleep,
            mock.patch.object(player, "stop_stream") as mock_stop_stream,
        ):
            player.stop()

        mock_wait_for_drain.assert_not_called()
        mock_sleep.assert_not_called()
        mock_stop_stream.assert_called_once_with()
        self.assertFalse(player.playing)

    def test_stop_closes_existing_stream_after_drain_stops_playback(self):
        player = AudioPlayer(sample_rate=24_000, buffer_size=2048)
        player.stream = mock.Mock()
        player.playing = False

        with mock.patch.object(player, "stop_stream") as mock_stop_stream:
            player.stop()

        mock_stop_stream.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
