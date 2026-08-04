from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.mmlifelong.adapter import build_mmlifelong_workspaces
from vcah.agent import VideoAgent
from vcah.captioning import (
    DEFAULT_CAPTION_PROMPT,
    CaptionGenerationConfig,
    OpenAICompatibleCaptionGenerator,
    run_caption_generation,
)
from vcah.caption_hybrid_search import CaptionHybridSearch
from vcah.caption_semantic_index import CaptionSemanticIndex
from vcah.caption_store import activate_caption_config
from vcah.embedding_adapter import SentenceTransformerEmbeddingAdapter
from vcah.evals import run_videomme_case
from vcah.model_client import OpenAICompatibleClient
from vcah.multiround import VirtualVideoMultiRoundDriver
from vcah.virtual_index import build_virtual_beat_index
from vcah.virtual_video import VirtualVideoWorkspace, materialize_lowfps_frame_cache
from vcah.videomme_virtual import build_videomme_smoke_workspaces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the slim evidence-seeking video agent.")
    subparsers = parser.add_subparsers(dest="command")
    vv_build = subparsers.add_parser(
        "vv-build-videomme",
        help="Build VideoMME virtual-video smoke workspaces.",
        description="Build three VideoMME virtual-video smoke workspaces.",
    )
    vv_build.add_argument("--dataset-root", required=True, help="VideoMME snapshot root containing videomme/, video/, subtitle/.")
    vv_build.add_argument("--out-dir", required=True, help="Output directory for per-case virtual workspaces.")
    vv_build.add_argument("--seed", type=int, default=20260707)

    mml_build = subparsers.add_parser(
        "vv-build-mmlifelong",
        help="Build shared MM-Lifelong Day virtual-video assets and cases.",
        description="Build the MM-Lifelong Day/game virtual timeline without creating a merged video.",
    )
    mml_build.add_argument("--dataset-root", required=True, help="MM-Lifelong root containing day/ and videos/day/.")
    mml_build.add_argument("--subset", choices=("game", "day"), default="game")
    mml_build.add_argument("--split", default="test")
    mml_build.add_argument("--asset-root", required=True, help="Shared timeline, frame-cache, and index directory.")
    mml_build.add_argument(
        "--case-root",
        required=True,
        help="Directory containing one lightweight workspace per case.",
    )
    mml_build.add_argument("--verify-durations", action=argparse.BooleanOptionalAction, default=True)
    mml_build.add_argument("--verify-clues", action=argparse.BooleanOptionalAction, default=True)
    mml_build.add_argument("--overwrite", action="store_true")

    vv_caption = subparsers.add_parser(
        "vv-caption",
        help="Generate resumable shared captions for one virtual-video asset.",
        description="Generate chunk captions and timestamp-aware passages without modifying ColdIndex.",
    )
    vv_caption.add_argument("--asset-root", required=True)
    vv_caption.add_argument("--config", required=True, help="OpenAI-compatible multimodal API YAML.")
    vv_caption.add_argument("--config-section", default="investigator_api")
    vv_caption.add_argument("--model", help="Optional assertion for the model configured in YAML.")
    vv_caption.add_argument("--provider", help="Provider label stored with caption provenance.")
    vv_caption.add_argument("--prompt-file")
    vv_caption.add_argument("--chunk-sec", type=float, default=300.0)
    vv_caption.add_argument("--sample-fps", type=float, default=1.0)
    vv_caption.add_argument("--max-frames", type=int, default=300)
    vv_caption.add_argument(
        "--frame-extraction-mode",
        choices=("seek", "fps_batch"),
        default="seek",
    )
    vv_caption.add_argument("--image-width", type=int)
    vv_caption.add_argument("--image-height", type=int)
    vv_caption.add_argument("--jpeg-quality", type=int)
    vv_caption.add_argument(
        "--append-timestamp-map",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    vv_caption.add_argument("--max-tokens", type=int, default=1800)
    vv_caption.add_argument("--max-retries", type=int, default=2)
    vv_caption.add_argument(
        "--timestamp-shift-mode",
        choices=(
            "deterministic",
            "deterministic_rema",
            "deterministic_rema_v2",
            "deterministic_rema_v3",
        ),
        default="deterministic",
    )
    vv_caption.add_argument("--max-chunks", type=int)
    vv_caption.add_argument("--start-chunk", type=int, default=0)
    vv_caption.add_argument("--workers", type=int, default=1)
    vv_caption.add_argument("--keep-frames", action=argparse.BooleanOptionalAction, default=False)
    vv_caption.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    caption_index = subparsers.add_parser(
        "vv-index-caption",
        help="Build a real dense or hybrid caption index.",
        description="Embed caption passages with sentence-transformers and persist an exact cosine index.",
    )
    caption_index.add_argument("--asset-root", required=True)
    caption_index.add_argument("--embedding-model", required=True)
    caption_index.add_argument("--embedding-revision")
    caption_index.add_argument("--device", default="cpu")
    caption_index.add_argument("--batch-size", type=int, default=64)
    caption_index.add_argument("--config-digest")
    caption_index.add_argument("--index-mode", choices=("dense", "hybrid"), default="hybrid")
    caption_index.add_argument("--rebuild", action="store_true")
    caption_index.add_argument("--activate", action=argparse.BooleanOptionalAction, default=True)

    vv_index = subparsers.add_parser(
        "vv-index",
        help="Build a virtual-video cold index.",
        description="Materialize low-fps frames and build beat thumbnail cold index for one workspace.",
    )
    vv_index.add_argument("--workspace", required=True, help="VirtualVideoWorkspace directory.")
    vv_index.add_argument("--low-fps", type=float, default=0.5)
    vv_index.add_argument("--beat-sec", type=float, default=18.0)

    vv_run = subparsers.add_parser(
        "vv-run",
        help="Run virtual-video multi-round investigation for one workspace.",
        description="Run Reasoner/Investigator multi-round loop for one VirtualVideoWorkspace.",
    )
    vv_run.add_argument("--workspace", required=True)
    vv_run.add_argument("--max-rounds", type=int, default=4)
    vv_run.add_argument("--max-investigations", type=int, default=20)

    vv_run_all = subparsers.add_parser(
        "vv-run-all",
        help="Run virtual-video multi-round investigation for all child workspaces.",
        description="Run all child directories containing case.json, including shared-asset workspaces.",
    )
    vv_run_all.add_argument("--workspace-root", required=True)
    vv_run_all.add_argument("--max-rounds", type=int, default=4)
    vv_run_all.add_argument("--max-investigations", type=int, default=20)
    parser.add_argument("--video", help="Path to a video file.")
    parser.add_argument("--question", help="Question to answer about the video.")
    parser.add_argument("--videomme-root", help="VideoMME root containing cases.json.")
    parser.add_argument("--case", help="VideoMME case id.")
    parser.add_argument("--run-dir", default="runs/default", help="Output directory for cold_index/ and run/.")
    parser.add_argument("--duration-sec", type=float, default=None, help="Optional duration override for smoke tests.")
    parser.add_argument("--index-mode", default="fast", help="Cold index mode label.")
    args = parser.parse_args()

    if args.command == "vv-build-videomme":
        workspaces = build_videomme_smoke_workspaces(Path(args.dataset_root), Path(args.out_dir), seed=args.seed)
        print(
            json.dumps(
                {
                    "workspace_count": len(workspaces),
                    "workspaces": [str(workspace.root_dir) for workspace in workspaces],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "vv-build-mmlifelong":
        result = build_mmlifelong_workspaces(
            Path(args.dataset_root),
            Path(args.asset_root),
            Path(args.case_root),
            subset=args.subset,
            split=args.split,
            verify_durations=args.verify_durations,
            verify_clues=args.verify_clues,
            overwrite=args.overwrite,
        )
        print(json.dumps(result.summary(), ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "vv-caption":
        api = OpenAICompatibleClient.from_yaml(Path(args.config), section=args.config_section)
        if args.model and str(args.model) != api.model:
            parser.error(f"--model {args.model!r} does not match configured model {api.model!r}")
        prompt = (
            Path(args.prompt_file).read_text(encoding="utf-8")
            if args.prompt_file
            else DEFAULT_CAPTION_PROMPT
        )
        config = CaptionGenerationConfig(
            model=api.model,
            provider=args.provider or api.api_type,
            prompt=prompt,
            chunk_sec=args.chunk_sec,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            frame_extraction_mode=args.frame_extraction_mode,
            image_width=args.image_width,
            image_height=args.image_height,
            jpeg_quality=args.jpeg_quality,
            append_timestamp_map=args.append_timestamp_map,
            timestamp_shift_mode=args.timestamp_shift_mode,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
        )
        generator = OpenAICompatibleCaptionGenerator(
            api,
            provider=config.provider,
            max_tokens=config.max_tokens,
        )
        result = run_caption_generation(
            Path(args.asset_root),
            config,
            generator,
            resume=args.resume,
            max_chunks=args.max_chunks,
            start_chunk=args.start_chunk,
            workers=args.workers,
            keep_frames=args.keep_frames,
        )
        print(json.dumps(result.summary(), ensure_ascii=False, indent=2, sort_keys=True))
        if result.failed_chunks:
            raise SystemExit(1)
        return

    if args.command == "vv-index-caption":
        adapter = SentenceTransformerEmbeddingAdapter(
            args.embedding_model,
            revision=args.embedding_revision,
            device=args.device,
            normalize=True,
            batch_size=args.batch_size,
        )
        dense = CaptionSemanticIndex.from_asset_root(
            Path(args.asset_root),
            adapter=adapter,
            config_digest=args.config_digest,
            rebuild=args.rebuild,
        )
        dense_manifest = dense.save_manifest(Path(args.asset_root))
        index = dense
        manifest = dense_manifest
        if args.index_mode == "hybrid":
            index = CaptionHybridSearch.from_asset_root(
                Path(args.asset_root),
                adapter=adapter,
                config_digest=dense.config_digest,
            )
            manifest = index.save_manifest(Path(args.asset_root))
        active_path = None
        if args.activate:
            active_path = activate_caption_config(
                Path(args.asset_root),
                index.config_digest,
                metadata={
                    "index_mode": args.index_mode,
                    "index_digest": index.index_digest,
                    "embedding_model": adapter.model_id,
                    "embedding_version": adapter.model_version,
                },
            )
        print(
            json.dumps(
                {
                    "asset_root": str(Path(args.asset_root)),
                    "index_mode": args.index_mode,
                    "index_digest": index.index_digest,
                    "config_digest": index.config_digest,
                    "passage_count": len(index.passages),
                    "embedding_model": adapter.model_id,
                    "embedding_version": adapter.model_version,
                    "embedding_dimension": adapter.dimension,
                    "normalize": adapter.normalize,
                    "manifest": str(manifest),
                    "active_config": str(active_path) if active_path else None,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "vv-index":
        workspace = VirtualVideoWorkspace.load(Path(args.workspace))
        frames = materialize_lowfps_frame_cache(workspace, fps=args.low_fps)
        result = build_virtual_beat_index(workspace, frames, beat_sec=args.beat_sec)
        print(
            json.dumps(
                {
                    "workspace": str(workspace.root_dir),
                    "frames": len(frames),
                    "beats": len(result.virtual_beats),
                    "cold_index": str(workspace.cold_index_dir),
                    "beat_index": str(result.beat_index_path),
                    "timeline_grid": str(result.timeline_grid_path),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "vv-run":
        workspace = VirtualVideoWorkspace.load(Path(args.workspace))
        result = VirtualVideoMultiRoundDriver(max_rounds=args.max_rounds, max_investigations=args.max_investigations).run(workspace)
        print(json.dumps(_vv_result_payload(result), ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "vv-run-all":
        root = Path(args.workspace_root)
        summaries = []
        for child in sorted(path for path in root.iterdir() if path.is_dir()):
            if not (child / "case.json").exists():
                continue
            workspace = VirtualVideoWorkspace.load(child)
            result = VirtualVideoMultiRoundDriver(max_rounds=args.max_rounds, max_investigations=args.max_investigations).run(workspace)
            summaries.append(_vv_result_payload(result))
        print(
            json.dumps(
                {
                    "case_count": len(summaries),
                    "correct": sum(1 for item in summaries if item["correct"]),
                    "cases": summaries,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    run_dir = Path(args.run_dir)
    if args.videomme_root and args.case:
        result = run_videomme_case(Path(args.videomme_root), args.case, run_dir=run_dir, duration_sec=args.duration_sec)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if not args.video or not args.question:
        parser.error("--video and --question are required unless --videomme-root and --case are provided")

    answer = VideoAgent().ask(
        args.video,
        args.question,
        run_dir=run_dir,
        duration_sec=args.duration_sec,
        index_mode=args.index_mode,
    )
    print(json.dumps({"answer": answer.answer, "citations": list(answer.citations)}, ensure_ascii=False, indent=2, sort_keys=True))


def _vv_result_payload(result) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "answer": result.answer,
        "citations": list(result.citations),
        "correct": result.correct,
        "verified": result.verified,
        "verification_reason": result.verification_reason,
        "rounds": result.rounds,
        "accepted_investigations": result.accepted_investigations,
    }


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_") or "case"


if __name__ == "__main__":
    main()
