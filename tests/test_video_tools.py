import unittest
from unittest import mock

from visual_coding_agent_harness.core.registry import ToolError
from visual_coding_agent_harness.legacy.tools.video_atomic import (
    build_extract_clip_command,
    build_sample_frames_command,
    require_ffmpeg,
)


class VideoAtomicToolsTest(unittest.TestCase):
    def test_build_sample_frames_command(self):
        command = build_sample_frames_command(
            video_path="input/movie.mp4",
            output_pattern="artifacts/frames/frame_%05d.jpg",
            fps=1.0,
            start_time=10.0,
            duration=5.0,
        )

        self.assertEqual(
            command,
            [
                "ffmpeg",
                "-y",
                "-ss",
                "10.0",
                "-t",
                "5.0",
                "-i",
                "input/movie.mp4",
                "-vf",
                "fps=1.0",
                "artifacts/frames/frame_%05d.jpg",
            ],
        )

    def test_build_extract_clip_command(self):
        command = build_extract_clip_command(
            video_path="input/movie.mp4",
            output_path="artifacts/clips/clip.mp4",
            start_time=12.5,
            duration=4.0,
        )

        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("input/movie.mp4", command)
        self.assertIn("artifacts/clips/clip.mp4", command)

    def test_require_ffmpeg_raises_clear_error_when_missing(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(ToolError) as context:
                require_ffmpeg()

            self.assertIn("ffmpeg is required", str(context.exception))


if __name__ == "__main__":
    unittest.main()
