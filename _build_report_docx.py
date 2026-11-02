"""生成《面单识别与匹配系统 项目报告》DOCX.

措辞原则:
  · 书面语, 避免口语与比喻
  · 结论以数据和事实支撑, 不作修饰性表述
  · 术语统一: 运单号 / 信息库 / 视觉模型 / 本系统
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = Path(r"D:\parcel_detective\parcel_detective_V2\项目报告书.docx")
VERSION_DIR = OUT.parent
DIST_DIR = VERSION_DIR / "dist" / "ParcelDetective"
PKG_FILE = VERSION_DIR / "dist" / "ParcelDetective_v2.1.0.7z"


def _dir_size_mb(p: Path) -> str:
    try:
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return f"{total / 1024 / 1024:.1f} MB"
    except Exception:                            # noqa: BLE001
        return "（未构建）"


def _file_size_mb(p: Path) -> str:
    try:
        return f"{p.stat().st_size / 1024 / 1024:.1f} MB"
    except Exception:                            # noqa: BLE001
        return "（未打包）"


APP_VERSION = "2.1.0"
DIR_SIZE = _dir_size_mb(DIST_DIR)
PKG_SIZE = _file_size_mb(PKG_FILE)

BODY_FONT_CN = "宋体"
BODY_FONT_EN = "Times New Roman"
HEAD_FONT_CN = "黑体"
MONO_FONT = "Consolas"
BODY_SIZE = Pt(10.5)
TABLE_SIZE = Pt(9)


# ---------------------------------------------------------------- 基础工具
def set_run_font(run, cn=BODY_FONT_CN, en=BODY_FONT_EN, size=None,
                 bold=None, color=None):
    run.font.name = en
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)      # 中文必须显式指定 eastAsia 字体
    if size is not None:
        run.font.size = size
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def set_col_widths(table, widths_cm: list[float]) -> None:
    """设定固定列宽.

    ⚠️ 只设 cell.width 是不够的: python-docx 不会同步更新 <w:tblGrid>,
    而 Word / LibreOffice 排版时以 tblGrid 为准, 结果表格会按默认等宽渲染,
    直接超出页面右边界。必须同时写 tblGrid、tblW, 并把 tblLayout 设为 fixed。
    """
    table.autofit = False
    table.allow_autofit = False
    tbl = table._tbl
    tblPr = tbl.tblPr

    total_tw = int(sum(widths_cm) / 2.54 * 1440)
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        tblPr.append(tblW)
    tblW.set(qn("w:type"), "dxa")
    tblW.set(qn("w:w"), str(total_tw))

    layout = tblPr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tblPr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = tbl.find(qn("w:tblGrid"))
    if grid is not None:
        cols = grid.findall(qn("w:gridCol"))
        for gc, w in zip(cols, widths_cm):
            gc.set(qn("w:w"), str(int(w / 2.54 * 1440)))

    for row in table.rows:
        for cell, w in zip(row.cells, widths_cm):
            cell.width = Cm(w)


def shade(cell, hex_color: str) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hex_color)
    tcpr.append(sh)


def para_border_bottom(paragraph) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:color"), "1F6FEB")
    bdr.append(bottom)
    ppr.append(bdr)


def add_page_number_footer(section) -> None:
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    set_run_font(run, size=Pt(9), color=RGBColor(0x60, 0x60, 0x60))
    f1 = OxmlElement("w:fldChar")
    f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = "PAGE"
    f2 = OxmlElement("w:fldChar")
    f2.set(qn("w:fldCharType"), "end")
    run._r.append(f1)
    run._r.append(it)
    run._r.append(f2)


# ---------------------------------------------------------------- 文档构件
doc = Document()

# 页面
sec = doc.sections[0]
sec.page_height = Cm(29.7)
sec.page_width = Cm(21.0)
sec.top_margin = Cm(2.4)
sec.bottom_margin = Cm(2.2)
sec.left_margin = Cm(2.6)
sec.right_margin = Cm(2.4)
add_page_number_footer(sec)

# 正文默认样式
normal = doc.styles["Normal"]
normal.font.name = BODY_FONT_EN
normal.font.size = BODY_SIZE
normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT_CN)
normal.paragraph_format.line_spacing = 1.45
normal.paragraph_format.space_after = Pt(6)

# 标题样式统一为黑体黑色
for lvl in (1, 2, 3):
    st = doc.styles[f"Heading {lvl}"]
    st.font.name = HEAD_FONT_CN
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.font.bold = True
    st.element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT_CN)
    st.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    st.paragraph_format.space_after = Pt(6)
    st.font.size = Pt(15) if lvl == 1 else (Pt(12) if lvl == 2 else Pt(11))


def H1(text: str) -> None:
    p = doc.add_heading(text, level=1)
    for r in p.runs:
        set_run_font(r, cn=HEAD_FONT_CN, en=HEAD_FONT_CN, size=Pt(15),
                     bold=True, color=RGBColor(0, 0, 0))
    para_border_bottom(p)


def H2(text: str) -> None:
    p = doc.add_heading(text, level=2)
    for r in p.runs:
        set_run_font(r, cn=HEAD_FONT_CN, en=HEAD_FONT_CN, size=Pt(12),
                     bold=True, color=RGBColor(0, 0, 0))


def P(text: str, *, indent: bool = True, size=None, italic=False) -> None:
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Pt(21)   # 首行缩进两字
    r = p.add_run(text)
    set_run_font(r, size=size or BODY_SIZE)
    if italic:
        r.font.italic = True


try:
    _lb = doc.styles["List Bullet"]
    _lb.font.name = BODY_FONT_EN
    _lb.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT_CN)
    _lb.font.size = BODY_SIZE
except Exception:
    pass


def BULLET(text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Pt(21)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    set_run_font(r, size=BODY_SIZE)


def CODE(lines: list[str]) -> None:
    """等宽代码/图示块, 用单元格做出底色与边框."""
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_col_widths(t, [16.0])
    cell = t.cell(0, 0)
    shade(cell, "F4F6F8")
    cell.paragraphs[0].text = ""
    first = True
    for ln in lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.05
        r = p.add_run(ln)
        set_run_font(r, cn=MONO_FONT, en=MONO_FONT, size=Pt(8.5))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def BLOCK_DIAGRAM() -> None:
    """架构块图.

    不用 ASCII 制表符: ─ │ ┌ 这类字符在中文字体下是**全角宽**,
    与等宽的英文字符混排时列宽不一致, 在 Word / LibreOffice 里会错位
    (实测出现"键检测库 一页"这类残缺)。改用表格 + 合并单元格表达层次,
    不依赖任何字符对齐。
    """
    W = [5.4, 5.3, 5.3]
    t = doc.add_table(rows=0, cols=3)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    def row_full(text: str, fill: str = "FFFFFF", bold: bool = False,
                 size=Pt(10)) -> None:
        cells = t.add_row().cells
        merged = cells[0].merge(cells[2])
        shade(merged, fill)
        pr = merged.paragraphs[0]
        pr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pr.paragraph_format.space_after = Pt(2)
        r = pr.add_run(text)
        set_run_font(r, cn=BODY_FONT_CN, en=BODY_FONT_CN, size=size, bold=bold)

    def row_cells(texts: list[str], fill: str = "FFFFFF",
                  size=Pt(9.5)) -> None:
        cells = t.add_row().cells
        for c, txt in zip(cells, texts):
            shade(c, fill)
            pr = c.paragraphs[0]
            pr.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pr.paragraph_format.space_after = Pt(2)
            pr.paragraph_format.line_spacing = 1.1
            r = pr.add_run(txt)
            set_run_font(r, cn=BODY_FONT_CN, en=BODY_FONT_CN, size=size)

    row_full("图形界面（PyQt5）：一键检测页　·　信息库页", "E8EEF7", True)
    row_full("▼", "FFFFFF", True, Pt(9))
    row_full("detect_pipeline　——　调度 · 判档 · 计时 · 服务预检", "E8EEF7", True)
    row_full("▼", "FFFFFF", True, Pt(9))
    row_cells(["ai_vision\n字段提取", "matcher\n加权匹配",
               "database\n数据持久化"], "F7F9FC")
    row_full("▼　　　　　　　　　　　　　▼", "FFFFFF", True, Pt(9))
    row_cells(["Ollama　本地视觉模型", "", "parcels.db（SQLite）"], "FBFBFB")

    set_col_widths(t, W)
    table_row_props(t)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def table_row_props(table) -> None:
    """排版惯例: 行内不跨页断开; 表头行跨页重复."""
    for i, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        if trPr.find(qn("w:cantSplit")) is None:
            trPr.append(OxmlElement("w:cantSplit"))
        if i == 0 and trPr.find(qn("w:tblHeader")) is None:
            trPr.append(OxmlElement("w:tblHeader"))


def TABLE(header: list[str], rows: list[list[str]],
          widths: list[float] | None = None,
          highlight_last_row: bool = False,
          caption: str | None = None) -> None:
    if caption:
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cp.add_run(caption)
        set_run_font(r, cn=HEAD_FONT_CN, en=HEAD_FONT_CN, size=Pt(9),
                     bold=True, color=RGBColor(0x44, 0x44, 0x44))
        cp.paragraph_format.space_after = Pt(3)
        cp.paragraph_format.keep_with_next = True   # 标题不单独留在上一页

    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = t.rows[0]
    for i, h in enumerate(header):
        c = hdr.cells[i]
        shade(c, "E8EEF7")
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraphs[0].paragraph_format.space_after = Pt(2)
        c.paragraphs[0].paragraph_format.line_spacing = 1.15
        r = c.paragraphs[0].add_run(h)
        set_run_font(r, cn=HEAD_FONT_CN, en=HEAD_FONT_CN, size=TABLE_SIZE,
                     bold=True)

    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        is_last = highlight_last_row and ri == len(rows) - 1
        for ci, val in enumerate(row):
            c = cells[ci]
            if is_last:
                shade(c, "F7F7F7")
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            p.alignment = (WD_ALIGN_PARAGRAPH.CENTER
                           if ci > 0 and len(str(val)) <= 14
                           else WD_ALIGN_PARAGRAPH.LEFT)
            r = p.add_run(str(val))
            set_run_font(r, size=TABLE_SIZE, bold=is_last)

    if widths:
        set_col_widths(t, widths)
    table_row_props(t)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ================================================================ 封面标题
tp = doc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
tp.paragraph_format.space_before = Pt(36)
tp.paragraph_format.space_after = Pt(4)
r = tp.add_run("面单识别与匹配系统")
set_run_font(r, cn=HEAD_FONT_CN, en=HEAD_FONT_CN, size=Pt(26), bold=True)

sp = doc.add_paragraph()
sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
sp.paragraph_format.space_after = Pt(24)
r = sp.add_run("项目报告")
set_run_font(r, cn=HEAD_FONT_CN, en=HEAD_FONT_CN, size=Pt(18), bold=True,
             color=RGBColor(0x1F, 0x6F, 0xEB))

TABLE(
    ["项目", "内容"],
    [["版本", APP_VERSION],
     ["报告性质", "阶段性状态报告"],
     ["编写立场", "项目负责人"],
     ["数据基准", "当前代码与实测结果，未经估算"],
     ["适用范围", "本报告描述系统当前状态，不涉及历史版本演进"]],
    widths=[3.6, 11.4],
)

doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ================================================================ 一
H1("一、项目定位")

P("本系统解决的是一个判定问题：一张面单进入系统后，应归属于信息库中的哪一条记录，"
  "以及是否可以不经人工直接放行。")

P("系统采用视觉大模型识别与多字段加权匹配相结合的技术路线，不依赖条码，"
  "亦不进行通用光学字符识别。该定位决定了以下三项基本特征：")

P("其一，输入为照片而非结构化数据。系统须容忍拍摄角度、光照、褶皱、遮挡与涂改。", indent=True)
P("其二，输出为判定结论而非纯文本。系统须给出「通过」「待确认」「人工」三档结论，"
  "并对该结论负责。", indent=True)
P("其三，系统具备兜底属性。其服务对象为自动化手段已经失效的异常件，"
  "识别失败属于设计预期之内的情形，不应视为故障。", indent=True)

# ================================================================ 二
H1("二、功能范围")

H2("2.1 已交付功能")

TABLE(
    ["功能", "载体", "状态"],
    [["单张与批量面单识别与匹配", "「一键检测」页", "已交付"],
     ["三档判定与处置建议", "「一键检测」页", "已交付"],
     ["图片对照（检测图与库中图并排）", "「一键检测」页", "已交付"],
     ["字段对照（识别值与库值逐项对应，悬停高亮）", "「一键检测」页", "已交付"],
     ["寄件方与收件方人工对调并重新判定", "「一键检测」页", "已交付"],
     ["信息库维护（增删改查、检索、破损标记）", "「信息库」页", "已交付"],
     ["面单图片内嵌预览与路径修复", "「信息库」页", "已交付"],
     ["JSON 导入、CSV 导出、自动备份", "「信息库」页", "已交付"],
     ["识别模型切换（3 个模型）与可用性自检", "「一键检测」页 / 命令行", "已交付"],
     ["命令行批处理与退出码约定", "程序入口", "已交付"]],
    widths=[8.4, 4.2, 2.4],
    caption="表 1　已交付功能清单",
)

H2("2.2 明确不在范围内的能力")

P("以下能力当前不具备。此项为设计决定，不属于缺陷：")

BULLET("条码识别。系统不采用该技术路线，面单条码损坏即视为该路信息不可用。")
BULLET("通用光学字符识别。系统不输出全文字符，仅输出约定的十五个字段。")
BULLET("并发处理。批量任务串行执行。")
BULLET("云端或联网推理。全部推理在本机完成。")

# ================================================================ 三
H1("三、技术方案")

H2("3.1 总体架构")

BLOCK_DIAGRAM()

P("各层职责界定明确：视觉识别模块仅负责读取，匹配模块仅负责比较，"
  "存储模块仅负责持久化，调度模块负责串联各环节并形成结论。"
  "三个核心模块之间不感知对方的内部实现。")

H2("3.2 识别方法")

P("系统调用本机 Ollama 服务上的视觉语言模型，通过固定提示词将输出约束为"
  "十五个约定字段的结构化结果。主要工程约束如下：")

BULLET("输出格式固定。采用结构化输出约束，避免模型自由表述。")
BULLET("温度参数取零，同一张图像重复识别应得到一致结果。")
BULLET("三级重试机制。首选结构化输出（上下文 32768）；失败时转普通 JSON 格式"
       "（上下文 65536）；再次失败则先缩小图像至 1200 像素后重试。"
       "该机制针对的是模型偶发不遵循输出格式的情形，而非识别准确率问题。")
BULLET("图像预缩放。长边超过阈值时先行缩放，以控制单张推理耗时。")
BULLET("不使用字面示例。提示词中不放置占位样例值。该约束源自实测中观察到的"
       "失效模式：模型会将示例值直接作为识别结果输出。")

H2("3.3 匹配方法")

P("采用分组加权与归一化的相似度模型，共设五个权重组：")

TABLE(
    ["权重组", "权重", "包含字段"],
    [["运单号", "0.40", "运单号（与订单号等价）"],
     ["收件电话", "0.20", "收件人电话"],
     ["收件地址", "0.20", "收件人地址（省／市／区／街道／详址分段）"],
     ["收件姓名", "0.10", "收件人姓名"],
     ["其它", "0.10", "快递公司、分拣码、品名、寄件方三项、到付金额、"
                      "保价金额、发货日期、寄件网点"]],
    widths=[2.6, 1.6, 11.8],
)

P("总分为各组相似度与归一化权重之积的求和。设计上包含三处主动处理：")

P("第一，权重动态重分配。输入缺失某一组时，其权重按比例转移至其余组，"
  "最后归一化至合计 1.0。目的在于使残缺输入仍可获得有判别意义的分数，"
  "避免因字段缺席而压低结果。")

P("第二，脱敏感知比对。信息库中的电话与姓名普遍带掩码。比对时按可见片段进行匹配，"
  "并施加完整度惩罚：仅命中一个尾号时，即使完全一致亦不给予满分。")

P("第三，证据量约束。单纯的归一化会导致仅含单个弱字段的输入获得满分——"
  "若模型仅识别出一个电话尾号，该字段的归一化权重即为 1.0，命中后总分为 1.0。"
  "在此类场景下等同于依据四位数字自动分拣包裹，风险不可接受。"
  "因此设定证据量下限，不足时强制降级为人工复核。")

P("此外，运单号与订单号被视为同一概念：任一有值即视为在场，"
  "比对时分别与库中两列比较并取较高分。")

H2("3.4 判定方法")

P("判定阈值集中于单一实现处，界面层不重复定义：")

CODE([
    "得分 ≥ 0.85 且字段有效         →  通过      自动打单或放行",
    "0.60 ≤ 得分 < 0.85            →  待确认    人工核验候选",
    "得分 < 0.60 或未提取到有效字段  →  人工      转人工处理",
])

P("判定之前先执行服务可用性预检（TCP 连接，超时 0.8 秒）。"
  "服务未启动时不进入推理流程，直接给出可操作的处置建议，"
  "以避免在无服务状态下空耗重试时间。")

H2("3.5 寄件方与收件方方向校正")

P("视觉模型存在将寄件方与收件方信息互换的系统性倾向。系统的处理方式是以信息库记录为裁判：")

P("分别计算「原样」与「对调」两个方向与库中记录的加权一致度。"
  "仅当对调后一致度提升不低于 0.15，且对调方向本身的可信度不低于 0.55 时，"
  "方执行对调。")

P("除自动校正外，界面提供人工对调入口。对调后按同一套阈值重新判定，"
  "以确保判定逻辑仅存在一处实现。")

H2("3.6 模型选择")

P("可选模型共三个：qwen3-vl:2b-instruct（默认）、qwen3-vl:4b-instruct-q4_k_m、"
  "qwen3-vl:8b-instruct-q4_k_m。清单在 core/model_registry.py 单点定义，"
  "界面下拉、命令行 --models 与 --ai-check --model 共用同一份数据。")

P("切换模型后立即检查 Ollama 服务状态与该模型是否已安装。检查把「服务不可用」"
  "与「模型未安装」分为两个独立结论，分别给出不同处置建议。检查失败时输出"
  "「模型可用性异常报告」并写入处理日志，程序不退出，用户可切回已安装模型继续工作。")

P("每次开始识别前复查一次，五分钟内的检查结果复用。模型不可用时取消本次识别并"
  "说明原因，避免在无模型状态下空耗三轮重试（单轮超时 90 秒）。"
  "模型选择不持久化，重启回到默认值；如需固定，使用环境变量 PARCEL_AI_MODEL "
  "或命令行 --model。")

P("本机仅安装 2B Instruct，4B 与 8B 未做精度实测；三者的清单、检查与传参链路"
  "已实现并有单元测试覆盖，但本报告不宣称更换模型后的准确率变化。")

# ================================================================ 四
H1("四、实现状况")

H2("4.1 交付物")

TABLE(
    ["项目", "内容"],
    [["版本", APP_VERSION],
     ["可执行程序", "ParcelDetective.exe（单目录部署）"],
     ["程序目录体积", DIR_SIZE],
     ["分发包体积", f"{PKG_SIZE}（7z 压缩）"],
     ["运行环境", "Python 3.11、PyQt5、OpenCV、Ollama，全部本地运行"],
     ["外部依赖", "本机 Ollama 服务及所选视觉模型"]],
    widths=[3.6, 11.4],
    caption="表 3　交付物清单",
)

H2("4.2 代码构成")

TABLE(
    ["模块", "行数", "职责"],
    [["core/matcher.py", "1468", "多字段相似度与加权匹配"],
     ["core/database.py", "1314", "数据持久化、导入导出、备份"],
     ["core/ai_vision.py", "1313", "视觉模型调用与字段提取"],
     ["core/model_registry.py", "307", "模型清单与可用性检查（v2.1.0 新增）"],
     ["core/detect_pipeline.py", "633", "调度、判定、计时、服务预检"],
     ["ui/detect_tab.py", "774", "一键检测页（含模型切换）"],
     ["ui/db_tab.py", "508", "信息库页"],
     ["ui/field_compare.py", "277", "字段对照组件"],
     ["ui/image_compare.py", "270", "图片对照组件"],
     ["main.py", "205", "程序入口与命令行"],
     ["ui/main_window.py", "130", "主窗口"],
     ["ui/ai_vision_ui.py", "71", "后台工作线程"],
     ["合计", "7270", ""]],
    widths=[5.4, 2.0, 7.6],
    highlight_last_row=True,
    caption="表 4　代码构成",
)

P("v2.1.0 未改动视觉识别、匹配、存储三个核心模块的识别与匹配逻辑；"
  "仅修正 ai_vision.check_service 的模型名比较方式（前缀匹配改为精确匹配），"
  "其余改动集中在模型注册表、检测页与命令行。")

H2("4.3 质量保证")

P("自动化测试共 450 项（332 个单元测试用例 + 118 项界面断言），分为六个层面：")

TABLE(
    ["测试范围", "项数", "侧重内容"],
    [["匹配逻辑", "99", "各字段相似度、权重归一化、约束边界"],
     ["数据持久化", "81", "增删改查、重复处理、导入导出、备份"],
     ["视觉识别", "73", "输出格式、异常重试、方向校正、真实服务联调"],
     ["处理流水线", "53", "判定边界、状态回调、服务不可用、异常不外抛、模型透传"],
     ["模型注册表", "26", "清单与默认值、名称归一化、前缀误判回归、异常报告"],
     ["界面", "118", "双页控件、结果导航、对照区刷新、缺图提示、模型切换与失败报告"]],
    widths=[3.4, 1.8, 9.8],
    caption="表 5　测试覆盖",
)

P("测试策略上有两项明确取舍：其一，界面测试通过真实控件驱动，而非直接调用内部函数，"
  "以覆盖「修改代码后未重新构建程序」一类问题；其二，失效用例标注为跳过而非删除，"
  "使能力缺口保持可见。")

H2("4.4 性能实测")

TABLE(
    ["指标", "实测值"],
    [["单张推理耗时（中位）", "4.8 秒"],
     ["单张推理耗时（平均）", "12.4 秒"],
     ["最快 / 最慢", "3.8 秒 / 276 秒"],
     ["服务未启动时的失败响应", "1.1 秒"]],
    widths=[6.4, 8.6],
    caption="表 6　性能指标",
)

P("平均耗时受少数异常样本显著抬高。实测中有三张图像耗时分别为 276 秒、192 秒与 109 秒，"
  "成因是三级重试各自等待满 90 秒超时。此为当前吞吐能力的薄弱环节。")

# ================================================================ 五
H1("五、数据基础")

H2("5.1 数据结构")

P("采用单表结构，表名 parcels，共 20 列，设三个索引（运单号、收件电话、创建时间）。"
  "字段覆盖运单号、订单号、收寄双方联系信息、分拣码、品名、金额、日期、网点、"
  "破损标记与图片路径。")

P("数据库默认位于用户数据目录，重建程序不覆盖用户数据；若程序同级存在数据库文件则优先使用，"
  "以支持整目录拷贝的便携用法。")

H2("5.2 数据质量现状")

P("此为本项目当前最需正视的环节。信息库现有 24 条记录，字段脱敏情况如下：")

TABLE(
    ["字段", "有值", "其中脱敏", "脱敏比例"],
    [["收件电话", "24", "22", "92%"],
     ["收件姓名", "23", "21", "91%"],
     ["品名", "21", "11", "52%"],
     ["寄件电话", "21", "8", "38%"],
     ["收件地址", "24", "1", "4%"],
     ["快递公司 / 分拣码", "24 / 24", "0 / 0", "0%"]],
    widths=[5.0, 3.0, 3.4, 3.6],
    caption="表 7　信息库字段脱敏情况",
)

P("由此产生两项后果，需予以明确说明：")

P("第一，脱敏导致区分度丧失。实测中存在多条不同记录共用同一脱敏电话的情形，"
  "该字段在比对时已不具备区分能力。")

P("第二，面单所印号码与库中所存号码并非同一数字。面单上印制的是隐私中转号"
  "（形如 17284046895转3142），信息库存储的是真实号码且已脱敏。二者天然不同，"
  "按号码比对无法命中。")

P("这构成一个数学意义上的上限：在现有数据条件下，即使运单号完全正确，"
  "加权总分的理论上限约为 0.61，低于 0.85 的放行门槛。")

# ================================================================ 六
H1("六、实测评估")

P("对全部 75 张真实面单完成完整评测，涵盖完好、揉捏破损、涂黑、物理遮挡四类样本。")

H2("6.1 判定分布")

TABLE(
    ["类别", "张数", "通过", "待确认", "人工", "读出运单号", "运单号在库"],
    [["完好面单", "29", "9", "11", "9", "26", "17"],
     ["揉捏破损", "6", "1", "0", "5", "6", "1"],
     ["涂黑面单", "29", "8", "10", "11", "28", "18"],
     ["物理遮挡", "11", "2", "4", "5", "11", "6"],
     ["合计", "75", "20", "25", "30", "71", "42"]],
    widths=[2.8, 1.7, 1.5, 1.8, 1.5, 2.3, 2.4],
    highlight_last_row=True,
    caption="表 8　判定分布",
)

H2("6.2 两项关键观测")

P("其一，图像质量并非主要瓶颈。完好面单通过率为 31%，涂黑面单为 28%，二者基本持平。"
  "若失分源于成像不清，涂黑样本应显著劣于完好样本。实测结果否定了这一假设。")

P("其二，识别能力本身可靠。运单号提取率为 94.7%（71/75）。"
  "更严格的验证来自配对检验：完好版与涂黑版为同一批 29 张面单的两个版本，"
  "理论上应读出完全相同的运单号。实测一致 20 对，不一致 8 对。")

P("不一致的 8 对暴露出一个明确的失效模式：模型会将隐私中转号或数字碎片填入运单号字段。"
  "示例如下：")

CODE([
    "完好版读出   18466326451转9651      实际为隐私中转号",
    "涂黑版读出   79040421472039         此为运单号",
])

P("此外还观测到单字符误读的情形，例如 772456317155299 与 777456317155299 仅一位之差。")

H2("6.3 三十张「人工」的责任归属")

TABLE(
    ["归因", "张数"],
    [["信息库中不存在该运单号（数据缺口）", "15"],
     ["模型将手机号或碎片填入运单号字段", "18"],
     ["运单号正确但其余字段无法对齐", "11"]],
    widths=[10.0, 5.0],
    caption="表 9　「人工」样本归因（存在重叠）",
)

P("据此可得出本次评测的核心结论：当前需要人工介入的样本，多数并非源于识别能力不足，"
  "而是数据基础与判分规则不相匹配。")

# ================================================================ 七
H1("七、已识别问题与风险")

TABLE(
    ["编号", "问题", "影响面", "性质"],
    [["P1", "加权打分的权重结构与脱敏数据不匹配，客观上无法达到放行门槛",
      "影响全部样本；25 张滞留待确认，11 张落入人工", "规则配置"],
     ["P2", "模型将隐私中转号或数字碎片误填为运单号", "18 张", "提示词与后处理"],
     ["P3", "信息库数据重度脱敏且已失去区分度", "全局性", "数据质量"],
     ["P4", "信息库缺少 15 个运单号", "15 张", "数据缺口"],
     ["P5", "个别样本耗时达 276 秒", "吞吐可预期性", "超时策略"]],
    widths=[1.4, 6.4, 5.4, 2.8],
    caption="表 10　问题与风险清单",
)

# ================================================================ 八
H1("八、后续工作")

P("以下为已形成方案的工作项。除特别说明者外，均待批准后执行。")

TABLE(
    ["优先级", "工作项", "预计改动量", "预期效果"],
    [["1", "放行判定改为「运单号精确且库中唯一即放行」，"
           "可叠加「另需一个字段达标」的缓解条件", "配置层面，约 1 行",
      "自动放行比例 26.7% 提升至 56.0%；需人工比例由 40.0% 降至 10.7%（基于实测数据推算）"],
     ["2", "为单张处理设置总时间预算（建议 60 秒）", "调度层，约 20 行",
      "最坏耗时由 276 秒降至 60 秒"],
     ["3", "运单号合法性校验与无效值丢弃（含「转」字、有效数字位不足、"
           "与电话号码相同）", "调度层，约 30 行",
      "部分改善 18 张误填样本，并使权重重分配机制正常生效"],
     ["4", "补录信息库缺失的 15 个运单号", "数据操作，不涉及代码", "该 15 张随即可用"],
     ["5", "信息库数据治理（引入未脱敏数据，或将脱敏字段从打分中剔除并重新分配权重）",
      "取决于数据来源", "从根本上改善加权打分的有效性"]],
    widths=[1.4, 6.6, 3.2, 4.8],
    caption="表 11　后续工作项",
)

P("前三项均在调度层与配置层完成，不涉及三个核心模块的逻辑改动，风险可控。")

# ================================================================ 九
H1("九、结论与建议")

P("技术判断：系统的识别能力已满足使用要求，当前瓶颈在于数据基础与判分规则。")

P("该判断有两组数据支撑：其一，运单号提取率为 94.7%；其二，完好与涂黑两类的通过率基本持平"
  "（31% 与 28%），可排除成像质量为主因的可能。")

P("建议按以下次序推进：")

P("第一，优先调整判分规则（工作项 1、2）。此项为纯配置与调度层改动，"
  "不需新增数据，亦不触及核心模块，预期可将人工介入量由四成降至一成左右。"
  "在现有条件下，其投入产出比最高。")

P("第二，并行推进数据补录（工作项 4）。该项不占用开发资源，由业务侧完成即可产生即时效果。")

P("第三，将数据治理列为中期任务（工作项 5）。信息库数据的脱敏程度决定了加权打分的理论上限。"
  "若长期无法取得未脱敏数据，则需重新设计打分结构，"
  "将判别力让渡给脱敏率较低的地址与分拣码字段。")

P("需补充说明：三档判定中的「待确认」一档，其设计意图为人工仅需快速确认，无需从头核对。"
  "本次评测中 25 张落入此档，其工作量性质与「人工」档存在实质差异，"
  "评估现场投入时宜分别计量。")

P("当前状态具备进入试用条件。建议试用期间同时启用调整后的判分规则，"
  "以实测数据替代推算数据验证效果。")

# ================================================================ 附
H1("附：文档索引")

TABLE(
    ["文档", "内容"],
    [["技术文档.md", "模块接口、数据模型、算法细节、构建说明、故障排查"],
     ["架构图.md", "总体结构、数据流、模块依赖、目录布局"],
     ["评测报告与改进方案.md", "75 张实测明细、根因分析、方案量化收益与风险"],
     ["records.jsonl", "逐张评测原始记录"]],
    widths=[5.0, 10.0],
    caption="表 12　相关文档",
)

doc.save(OUT)
print(f"  已生成: {OUT}")
print(f"  大小: {OUT.stat().st_size / 1024:.1f} KB")
