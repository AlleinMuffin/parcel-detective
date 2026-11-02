"""截取「信息库」页 —— 用有代表性的示例数据填充.

之前用空库截图(状态栏 0 条记录), 页面是空的, 没有佐证价值。
这里灌入 12 条贴近真实的数据: 脱敏电话/姓名、破损标记、一条缺图记录,
并选中一行以带出右下角的图片预览。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

V2 = Path(r"D:\parcel_detective\parcel_detective_V2\src")
sys.path.insert(0, str(V2))

from PyQt5.QtWidgets import QApplication      # noqa: E402

from core import database as db               # noqa: E402
from ui.main_window import MainWindow         # noqa: E402

OUT = Path(r"D:\parcel_detective\parcel_detective_V2\_shots")
OUT.mkdir(exist_ok=True)
PIC = Path(r"D:\parcel_detective\pic_data\面单包裹(1)\面单包裹")

#: 12 条示例记录 (脱敏写法与真实信息库一致)
ROWS = [
    dict(tracking_number="9817883403197", recipient_name="曾*",
         recipient_phone="*******2535",
         recipient_address="广东省茂名市化州市平定镇那宾生活超市",
         sender_name="覃**", sender_phone="138****5385",
         sender_address="广东省佛山市南海区狮山镇大学城",
         sorting_code="194-194-Y17-003", order_number="W202609228816",
         item_name="鞋舌贴", cod_amount="", insured_amount="",
         shipping_date="2026-09-22", courier_company="中国邮政EMS",
         sender_branch="化州平定", is_damaged=0, damage_type="",
         folder="完好面单", img="new (1)zep.jpg"),
    dict(tracking_number="YT0718248356076", recipient_name="官*",
         recipient_phone="*******4343",
         recipient_address="广西壮族自治区南宁市青秀区民族大道 128 号",
         sender_name="李**", sender_phone="1******0102",
         sender_address="广东省广州市白云区太和镇",
         sorting_code="530-001-A02-011", order_number="YTW2026092301",
         item_name="服装", cod_amount="0", insured_amount="200",
         shipping_date="2026-09-23", courier_company="圆通速递",
         sender_branch="广州太和", is_damaged=1, damage_type="外包装挤压",
         folder="揉捏波坏班", img=None),
    dict(tracking_number="777456317155299", recipient_name="田*",
         recipient_phone="*******8771",
         recipient_address="湖南省长沙市岳麓区麓山南路 932 号",
         sender_name="张**", sender_phone="1******0060",
         sender_address="湖北省武汉市洪山区珞喻路",
         sorting_code="410-012-B07-002", order_number="",
         item_name="电子配件", cod_amount="", insured_amount="",
         shipping_date="2026-09-24", courier_company="中国邮政EMS",
         sender_branch="长沙岳麓", is_damaged=0, damage_type="",
         folder="完好面单", img="new (10)lzd.jpg"),
    dict(tracking_number="JT3180192749755", recipient_name="魏丽艳",
         recipient_phone="18054201419",
         recipient_address="浙江省杭州市余杭区仓前街道文一西路 969 号",
         sender_name="王**", sender_phone="1******7210",
         sender_address="江苏省南京市江宁区将军大道",
         sorting_code="311-100-C03-018", order_number="JD0261189133871",
         item_name="日用百货", cod_amount="", insured_amount="500",
         shipping_date="2026-09-25", courier_company="极兔速递",
         sender_branch="杭州仓前", is_damaged=0, damage_type="",
         folder="完好面单", img="new (2).jpg"),
    dict(tracking_number="770345615390879", recipient_name="黄佳煌",
         recipient_phone="18998457673",
         recipient_address="福建省厦门市集美区杏林湾路 3399 号",
         sender_name="陈**", sender_phone="1******3344",
         sender_address="福建省泉州市晋江市陈埭镇",
         sorting_code="361-021-D05-007", order_number="W202609261102",
         item_name="鞋材", cod_amount="88", insured_amount="",
         shipping_date="2026-09-26", courier_company="申通快递",
         sender_branch="厦门集美", is_damaged=0, damage_type="",
         folder="完好面单", img="new (3).jpg"),
    dict(tracking_number="76978885290004", recipient_name="王信志",
         recipient_phone="18939847935",
         recipient_address="四川省成都市武侯区天府大道北段 1700 号",
         sender_name="赵**", sender_phone="1******8899",
         sender_address="重庆市渝北区龙溪街道",
         sorting_code="610-031-E08-014", order_number="",
         item_name="五金件", cod_amount="", insured_amount="1000",
         shipping_date="2026-09-27", courier_company="中通快递",
         sender_branch="成都武侯", is_damaged=0, damage_type="",
         folder="完好面单", img="new (5).jpg"),
    dict(tracking_number="79139967559982", recipient_name="刘*",
         recipient_phone="*******1201",
         recipient_address="山东省青岛市市南区香港中路 76 号",
         sender_name="孙**", sender_phone="1******5566",
         sender_address="山东省潍坊市奎文区东风东街",
         sorting_code="266-071-F02-009", order_number="W202609281230",
         item_name="纺织品", cod_amount="", insured_amount="",
         shipping_date="2026-09-28", courier_company="韵达速递",
         sender_branch="青岛市南", is_damaged=1, damage_type="面单涂黑",
         folder="涂黑面单", img="new (6).jpg"),
    dict(tracking_number="7915046100030", recipient_name="周*",
         recipient_phone="*******6642",
         recipient_address="河南省郑州市金水区农业路 100 号",
         sender_name="吴**", sender_phone="1******2233",
         sender_address="河南省洛阳市涧西区南昌路",
         sorting_code="450-008-G11-003", order_number="",
         item_name="食品", cod_amount="", insured_amount="",
         shipping_date="2026-09-29", courier_company="中国邮政EMS",
         sender_branch="郑州金水", is_damaged=0, damage_type="",
         folder="完好面单", img="new (7).jpg"),
    dict(tracking_number="9819989866891", recipient_name="郑*",
         recipient_phone="*******3391",
         recipient_address="陕西省西安市雁塔区长安南路 199 号",
         sender_name="冯**", sender_phone="1******7788",
         sender_address="陕西省咸阳市秦都区人民路",
         sorting_code="710-061-H06-012", order_number="W202609291401",
         item_name="文具", cod_amount="", insured_amount="",
         shipping_date="2026-09-29", courier_company="圆通速递",
         sender_branch="西安雁塔", is_damaged=0, damage_type="",
         folder="完好面单", img="new (8).jpg"),
    dict(tracking_number="79040421472039", recipient_name="杨*",
         recipient_phone="*******7712",
         recipient_address="辽宁省沈阳市和平区青年大街 286 号",
         sender_name="许**", sender_phone="1******4411",
         sender_address="辽宁省大连市中山区人民路",
         sorting_code="110-003-J09-021", order_number="JD0261189133871",
         item_name="机械配件", cod_amount="", insured_amount="",
         shipping_date="2026-09-30", courier_company="京东物流",
         sender_branch="沈阳和平", is_damaged=0, damage_type="",
         folder="物理遮挡版", img="微信图片_2026-10-23_164758_638.jpg"),
    dict(tracking_number="79145190061768", recipient_name="何*",
         recipient_phone="*******2288",
         recipient_address="安徽省合肥市蜀山区黄山路 443 号",
         sender_name="邓**", sender_phone="1******9900",
         sender_address="安徽省芜湖市镜湖区北京路",
         sorting_code="230-031-K04-016", order_number="",
         item_name="家用电器", cod_amount="", insured_amount="2000",
         shipping_date="2026-10-01", courier_company="中国邮政EMS",
         sender_branch="合肥蜀山", is_damaged=0, damage_type="",
         folder="完好面单", img="new (9)lzd.jpg"),
    # 故意留一条缺图记录, 体现"修复图片路径"的使用场景
    dict(tracking_number="7904021472039", recipient_name="罗*",
         recipient_phone="*******5507",
         recipient_address="云南省昆明市五华区一二一大街 134 号",
         sender_name="梁**", sender_phone="1******6622",
         sender_address="云南省曲靖市麒麟区南宁东路",
         sorting_code="650-022-L07-005", order_number="W202610020933",
         item_name="茶叶", cod_amount="", insured_amount="",
         shipping_date="2026-10-02", courier_company="中通快递",
         sender_branch="昆明五华", is_damaged=0, damage_type="",
         folder=None, img=None),
]


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    d = tempfile.TemporaryDirectory()
    dbp = Path(d.name) / "demo.db"
    db.init_database(dbp)

    n_img = 0
    for r in ROWS:
        rec = {k: v for k, v in r.items() if k not in ("folder", "img")}
        folder, img = r.get("folder"), r.get("img")
        if folder and img:
            fp = PIC / folder / img
            if fp.is_file():
                rec["image_path"] = str(fp)
                n_img += 1
            else:
                rec["image_path"] = ""
        else:
            # 缺图场景: 路径非空但文件不存在
            rec["image_path"] = str(PIC / "完好面单" / "已删除的图片.jpg")
        db.save_record(rec, path=dbp)

    win = MainWindow(db_path=dbp)
    win.resize(1600, 1000)
    win.show()
    app.processEvents()

    win.tabs.setCurrentIndex(1)          # 切到「信息库」
    app.processEvents()
    tab = win.db_tab
    tab.refresh() if hasattr(tab, "refresh") else None
    app.processEvents()

    # 选中一条**有图**的记录, 让右下角预览显示面单照片;
    # 选缺图那条会得到一片空白, 反而不利佐证。
    target = "9817883403197"
    hit = -1
    for r in range(tab.table.rowCount()):
        if tab.table.item(r, 1) and target in (tab.table.item(r, 1).text() or ""):
            hit = r
            break
    if hit < 0:
        hit = 0
    tab.table.selectRow(hit)
    app.processEvents()
    tab.table.scrollToItem(tab.table.item(hit, 1))
    app.processEvents()

    p = OUT / "ui-信息库.png"
    win.grab().save(str(p))
    print(f"  已保存: {p}  ({p.stat().st_size/1024:.0f} KB)")
    print(f"  记录数: {tab.table.rowCount()}   带图: {n_img}   缺图: 1")

    win.close()
    d.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
