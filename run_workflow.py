from report_writer.config import load_settings
from report_writer.service import ReportWriterService

settings = load_settings()
service = ReportWriterService(settings)

# Step 1: 摄入资料
print("=" * 60)
print("Step 1: Ingesting documents...")
manifest = service.ingest_folder(
    project_name="能源系统鲁棒优化综述",
    source_dir="资料",
    user_request="撰写一篇关于能源系统鲁棒优化与分布鲁棒优化的综述论文，包括模型方法、求解算法、应用场景等",
    target_words=20000,
)
print(f"Project created: {manifest.project_id}")
print(f"Documents: {manifest.document_ids}")
print(f"Chunks: {manifest.chunk_count}")

# Step 2: 生成大纲
print("\n" + "=" * 60)
print("Step 2: Generating outline...")
manifest = service.generate_outline(
    manifest.project_id,
    title_hint="能源系统鲁棒优化与分布鲁棒优化方法综述",
    additional_requirements="涵盖两阶段鲁棒优化、分布鲁棒优化在综合能源系统中的应用，包括模型、求解算法、性能评估",
)
print(f"Outline generated with {len(manifest.outline)} sections")
for s in manifest.outline:
    print(f"  {s.node_id}. {s.title}")

# Step 3: 生成报告
print("\n" + "=" * 60)
print("Step 3: Generating report (this may take a while)...")
report = service.generate_report(
    manifest.project_id,
    title_hint="能源系统鲁棒优化与分布鲁棒优化方法综述",
    additional_requirements="",
    target_words=20000,
)
print(f"Report generated: {report.report_id}")
print(f"Word count: {report.word_count}")
print(f"Exported files: {list(report.files.keys())}")

# Save report ID for later
with open("last_report_id.txt", "w") as f:
    f.write(report.report_id)
print(f"\nReport ID saved to last_report_id.txt: {report.report_id}")

service.close()