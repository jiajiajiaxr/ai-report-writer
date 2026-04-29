from __future__ import annotations

import argparse
import json
import sys

from .config import load_settings
from .service import ReportWriterService
from .siliconflow import SiliconFlowHTTPStatusError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI 写报告工具 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="导入资料并构建知识底座")
    ingest.add_argument("--project-name", required=True)
    ingest.add_argument("--source-dir", required=True)
    ingest.add_argument("--user-request", required=True)
    ingest.add_argument("--target-words", type=int, default=12000)

    outline = subparsers.add_parser("outline", help="生成大纲")
    outline.add_argument("--project-id", required=True)
    outline.add_argument("--title-hint")
    outline.add_argument("--additional-requirements")

    generate = subparsers.add_parser("generate", help="生成报告")
    generate.add_argument("--project-id", required=True)
    generate.add_argument("--title-hint")
    generate.add_argument("--additional-requirements")
    generate.add_argument("--target-words", type=int)

    revise = subparsers.add_parser("revise", help="修订报告")
    revise.add_argument("--project-id", required=True)
    revise.add_argument("--report-id")
    revise.add_argument("--revision-request", required=True)

    review = subparsers.add_parser("review", help="运行审查智能体")
    review.add_argument("--project-id", required=True)
    review.add_argument("--report-id")

    review_loop = subparsers.add_parser("review-loop", help="自动复审直到通过或达到轮数上限")
    review_loop.add_argument("--project-id", required=True)
    review_loop.add_argument("--report-id")
    review_loop.add_argument("--title-hint")
    review_loop.add_argument("--additional-requirements")
    review_loop.add_argument("--target-words", type=int)
    review_loop.add_argument("--max-rounds", type=int, default=3)

    subparsers.add_parser("check-api", help="检查 SiliconFlow 连通性")

    show = subparsers.add_parser("show", help="查看项目摘要")
    show.add_argument("--project-id", required=True)

    return parser


def _friendly_error(exc: Exception) -> str | None:
    if isinstance(exc, SiliconFlowHTTPStatusError):
        if exc.is_balance_insufficient:
            return "SiliconFlow API 调用失败：账户余额不足（HTTP 403 / code 30001）。请充值或更换可用 key 后重试。"
        if exc.is_rate_limited:
            return "SiliconFlow API 调用失败：当前触发限流，请稍后重试。"
        return str(exc)
    if isinstance(exc, (RuntimeError, ValueError)):
        return str(exc)
    return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    service = ReportWriterService(load_settings())
    try:
        try:
            if args.command == "ingest":
                result = service.ingest_folder(
                    project_name=args.project_name,
                    source_dir=args.source_dir,
                    user_request=args.user_request,
                    target_words=args.target_words,
                )
            elif args.command == "outline":
                result = service.generate_outline(
                    args.project_id,
                    title_hint=args.title_hint,
                    additional_requirements=args.additional_requirements,
                )
            elif args.command == "generate":
                result = service.generate_report(
                    args.project_id,
                    title_hint=args.title_hint,
                    additional_requirements=args.additional_requirements,
                    target_words=args.target_words,
                )
            elif args.command == "revise":
                result = service.revise_report(
                    args.project_id,
                    args.revision_request,
                    report_id=args.report_id,
                )
            elif args.command == "review":
                result = service.review_report(args.project_id, args.report_id)
            elif args.command == "review-loop":
                result = service.review_until_pass(
                    args.project_id,
                    report_id=args.report_id,
                    title_hint=args.title_hint,
                    additional_requirements=args.additional_requirements,
                    target_words=args.target_words,
                    max_rounds=args.max_rounds,
                )
            elif args.command == "check-api":
                result = service.check_api()
            elif args.command == "show":
                result = service.get_project(args.project_id)
            else:
                raise ValueError(f"Unknown command: {args.command}")
        except Exception as exc:
            message = _friendly_error(exc)
            if message:
                print(message, file=sys.stderr)
                raise SystemExit(1) from None
            raise

        if hasattr(result, "model_dump_json"):
            print(result.model_dump_json(indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
