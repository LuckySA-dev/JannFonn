"""
app.py  (v2.2)
--------------
Streamlit UI สำหรับระบบแนะนำเส้นทางท่องเที่ยว

เริ่มใช้งาน:
    streamlit run app.py

การปรับปรุง v2.2
----------------
- แก้บัก refresh: บันทึกกราฟลงไฟล์ graph_data.json ทุกครั้งที่แก้ไข
- แก้บัก isolated node: node ที่ยังไม่มีเส้นเชื่อมวางแถวล่าง ไม่กระเจิง
- เพิ่ม: Search → node ที่หาเจอไฮไลต์สีแดงในกราฟ
- ลบ: ตารางเปรียบเทียบ Insertion vs Merge Sort
- คง: CRUD ครบ, TSP, สถิติกราฟ, ตารางเส้นทาง
"""

import json
import os
import streamlit as st
import pandas as pd
import networkx as nx

from graph_manager import GraphManager
from algorithms import (
    merge_sort,
    sequential_search,
    tsp_nearest_neighbor,
)
from visualization import draw_graph


# ===========================================================================
# Page configuration
# ===========================================================================

st.set_page_config(
    page_title="ระบบแนะนำเส้นทางท่องเที่ยว",
    page_icon="🗺️",
    layout="wide",
)

st.markdown(
    "<style>#MainMenu{visibility:hidden}footer{visibility:hidden}</style>",
    unsafe_allow_html=True,
)


# ===========================================================================
# File persistence — บันทึกกราฟลงดิสก์ ไม่หายเมื่อรีเฟรช
# ===========================================================================

_DATA_FILE = "graph_data.json"


def _load_from_file() -> dict | None:
    if os.path.exists(_DATA_FILE):
        try:
            with open(_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_to_file(data: dict) -> None:
    with open(_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ===========================================================================
# ข้อมูลตัวอย่าง (Paris)
# ===========================================================================

_SAMPLE_LOCATIONS = [
    "หอไอเฟล", "พิพิธภัณฑ์ลูฟว์", "มหาวิหารนอเทรอดาม",
    "พระราชวังแวร์ซาย", "ประตูชัยปารีส", "มงมาร์ต", "พิพิธภัณฑ์ออร์แซ",
]

_SAMPLE_PATHS = [
    ("หอไอเฟล",            "พิพิธภัณฑ์ลูฟว์",        3.5),
    ("หอไอเฟล",            "พิพิธภัณฑ์ออร์แซ",        1.2),
    ("หอไอเฟล",            "ประตูชัยปารีส",            2.8),
    ("พิพิธภัณฑ์ลูฟว์",    "มหาวิหารนอเทรอดาม",       4.0),
    ("พิพิธภัณฑ์ลูฟว์",    "ประตูชัยปารีส",            3.3),
    ("พิพิธภัณฑ์ลูฟว์",    "มงมาร์ต",                  4.5),
    ("มหาวิหารนอเทรอดาม",  "พิพิธภัณฑ์ออร์แซ",        2.1),
    ("มหาวิหารนอเทรอดาม",  "พระราชวังแวร์ซาย",        19.5),
    ("พระราชวังแวร์ซาย",   "ประตูชัยปารีส",           18.0),
    ("ประตูชัยปารีส",       "มงมาร์ต",                  2.6),
    ("มงมาร์ต",             "พิพิธภัณฑ์ลูฟว์",          4.5),
    ("พิพิธภัณฑ์ออร์แซ",   "พิพิธภัณฑ์ลูฟว์",          2.9),
]


def _build_sample_graph() -> GraphManager:
    gm = GraphManager()
    for loc in _SAMPLE_LOCATIONS:
        gm.add_location(loc)
    for loc1, loc2, dist in _SAMPLE_PATHS:
        gm.add_path(loc1, loc2, dist)
    return gm


# ===========================================================================
# Session State
# ===========================================================================

def init_session_state() -> None:
    if "graph_data" not in st.session_state:
        # โหลดจากไฟล์ก่อน ถ้าไม่มีค่อยใช้ sample
        from_file = _load_from_file()
        st.session_state.graph_data = from_file if from_file else _build_sample_graph().to_dict()

    if "optimal_route"  not in st.session_state:
        st.session_state.optimal_route  = None
    if "total_distance" not in st.session_state:
        st.session_state.total_distance = None
    if "route_error"    not in st.session_state:
        st.session_state.route_error    = None
    if "route_speed"    not in st.session_state:
        st.session_state.route_speed    = 40.0
    if "search_results" not in st.session_state:
        st.session_state.search_results = None


def get_graph_manager() -> GraphManager:
    return GraphManager.from_dict(st.session_state.graph_data)


def save_and_rerun(gm: GraphManager, msg: str) -> None:
    """บันทึก → เขียนไฟล์ → ล้าง route เก่า → rerun (แก้ stale data)"""
    data = gm.to_dict()
    st.session_state.graph_data     = data
    st.session_state.optimal_route  = None
    st.session_state.total_distance = None
    st.session_state.route_error    = None
    if "multiselect_locations" in st.session_state:
        del st.session_state["multiselect_locations"]
    _save_to_file(data)   # persist ลงดิสก์
    st.toast(msg, icon="✅")
    st.rerun()


# ===========================================================================
# Sidebar
# ===========================================================================

def render_sidebar() -> None:

    st.sidebar.markdown(
        "<div style='background:linear-gradient(135deg,#1565C0,#00695C);"
        "padding:14px;border-radius:10px;text-align:center;color:white;"
        "font-size:18px;font-weight:bold;margin-bottom:8px'>"
        "🗺️ ระบบแนะนำเส้นทาง<br>"
        "<span style='font-size:12px;font-weight:normal'>"
        "Tourism Route Planner v2</span></div>",
        unsafe_allow_html=True,
    )

    gm        = get_graph_manager()
    locations = gm.get_locations()
    paths     = gm.get_paths()

    # -----------------------------------------------------------------------
    # 1. เพิ่มสถานที่
    # -----------------------------------------------------------------------
    st.sidebar.subheader("➕ เพิ่มสถานที่")
    new_location = st.sidebar.text_input(
        "ชื่อสถานที่", key="input_add_location", placeholder="เช่น หอไอเฟล"
    )
    if st.sidebar.button("เพิ่มสถานที่", use_container_width=True, key="btn_add_loc"):
        if new_location.strip():
            ok, msg = gm.add_location(new_location)
            if ok:
                save_and_rerun(gm, msg)
            else:
                st.sidebar.error(msg)
        else:
            st.sidebar.warning("⚠️ กรุณากรอกชื่อสถานที่")

    st.sidebar.markdown("---")

    # -----------------------------------------------------------------------
    # 2. ลบสถานที่
    # -----------------------------------------------------------------------
    st.sidebar.subheader("🗑️ ลบสถานที่")
    if locations:
        loc_to_remove = st.sidebar.selectbox(
            "เลือกสถานที่ที่ต้องการลบ", options=locations, key="select_remove_location"
        )
        if st.sidebar.button("ลบสถานที่", use_container_width=True, key="btn_remove_loc"):
            ok, msg = gm.remove_location(loc_to_remove)
            if ok:
                save_and_rerun(gm, msg)
            else:
                st.sidebar.error(msg)
    else:
        st.sidebar.info("ยังไม่มีสถานที่ในระบบ")

    st.sidebar.markdown("---")

    # -----------------------------------------------------------------------
    # 3. เพิ่มเส้นทาง
    # -----------------------------------------------------------------------
    st.sidebar.subheader("🔗 เพิ่มเส้นทาง")
    if len(locations) >= 2:
        path_from = st.sidebar.selectbox(
            "จากสถานที่", options=locations, key="select_path_from"
        )
        to_options = [loc for loc in locations if loc != path_from]
        path_to    = st.sidebar.selectbox(
            "ไปยังสถานที่", options=to_options, key="select_path_to"
        )
        distance = st.sidebar.number_input(
            "ระยะทาง (กม.)", min_value=0.1, max_value=50000.0,
            value=10.0, step=0.1, key="input_distance",
        )
        if st.sidebar.button("เพิ่มเส้นทาง", use_container_width=True, key="btn_add_path"):
            ok, msg = gm.add_path(path_from, path_to, distance)
            if ok:
                save_and_rerun(gm, msg)
            else:
                st.sidebar.error(msg)
    else:
        st.sidebar.info("กรุณาเพิ่มสถานที่อย่างน้อย 2 แห่งก่อนสร้างเส้นทาง")

    st.sidebar.markdown("---")

    # -----------------------------------------------------------------------
    # 4. แก้ไขระยะทาง
    # -----------------------------------------------------------------------
    st.sidebar.subheader("✏️ แก้ไขระยะทาง")
    if paths:
        edit_labels = [
            f"{p['from']} ↔ {p['to']} ({p['distance']} กม.)" for p in paths
        ]
        edit_choice = st.sidebar.selectbox(
            "เลือกเส้นทางที่ต้องการแก้ไข", options=edit_labels, key="select_edit_path"
        )
        edit_idx  = edit_labels.index(edit_choice)
        edit_path = paths[edit_idx]
        new_dist  = st.sidebar.number_input(
            "ระยะทางใหม่ (กม.)", min_value=0.1, max_value=50000.0,
            value=float(edit_path["distance"]), step=0.1, key="input_edit_distance",
        )
        if st.sidebar.button("อัปเดตระยะทาง", use_container_width=True, key="btn_edit_path"):
            ok, msg = gm.update_path(edit_path["from"], edit_path["to"], new_dist)
            if ok:
                save_and_rerun(gm, msg)
            else:
                st.sidebar.error(msg)
    else:
        st.sidebar.info("ยังไม่มีเส้นทางในระบบ")

    st.sidebar.markdown("---")

    # -----------------------------------------------------------------------
    # 5. ลบเส้นทาง
    # -----------------------------------------------------------------------
    st.sidebar.subheader("✂️ ลบเส้นทาง")
    if paths:
        path_labels = [
            f"{p['from']} ↔ {p['to']} ({p['distance']} กม.)" for p in paths
        ]
        path_choice = st.sidebar.selectbox(
            "เลือกเส้นทางที่ต้องการลบ", options=path_labels, key="select_remove_path"
        )
        if st.sidebar.button("ลบเส้นทาง", use_container_width=True, key="btn_remove_path"):
            idx    = path_labels.index(path_choice)
            chosen = paths[idx]
            ok, msg = gm.remove_path(chosen["from"], chosen["to"])
            if ok:
                save_and_rerun(gm, msg)
            else:
                st.sidebar.error(msg)
    else:
        st.sidebar.info("ยังไม่มีเส้นทางในระบบ")

    st.sidebar.markdown("---")

    # -----------------------------------------------------------------------
    # 6. ค้นหาสถานที่ — Sequential Search (partial match)
    # -----------------------------------------------------------------------
    st.sidebar.subheader("🔍 ค้นหาสถานที่")
    query = st.sidebar.text_input(
        "กรอกชื่อสถานที่", key="input_search_query", placeholder="เช่น หอไอเฟล"
    )
    if st.sidebar.button("ค้นหา", use_container_width=True, key="btn_search"):
        if query.strip():
            locs = gm.get_locations()
            if not locs:
                st.sidebar.warning("⚠️ ยังไม่มีสถานที่ในระบบ")
            else:
                _, matches = sequential_search(locs, query)
                st.session_state.search_results = {
                    "query":   query,
                    "matches": matches,
                }
        else:
            st.sidebar.warning("⚠️ กรุณากรอกคำค้นหา")

    if st.session_state.search_results:
        sr = st.session_state.search_results
        st.sidebar.markdown(f"ผลการค้นหา **'{sr['query']}'**:")
        if sr["matches"]:
            for m in sr["matches"]:
                st.sidebar.markdown(f"  🔴 {m}")
        else:
            st.sidebar.info("ไม่พบสถานที่ที่ตรงกับคำค้นหา")
        if st.sidebar.button("ล้างผลค้นหา", key="btn_clear_search", use_container_width=True):
            st.session_state.search_results = None
            st.rerun()

    st.sidebar.markdown("---")

    # -----------------------------------------------------------------------
    # 7. รีเซ็ตข้อมูล
    # -----------------------------------------------------------------------
    st.sidebar.subheader("🔄 รีเซ็ตข้อมูล")
    if st.sidebar.button(
        "🔄 รีเซ็ตเป็นข้อมูลตัวอย่าง",
        use_container_width=True, key="btn_reset",
    ):
        # ลบไฟล์ด้วยเพื่อให้ reload ไปใช้ sample
        if os.path.exists(_DATA_FILE):
            os.remove(_DATA_FILE)
        for k in [
            "graph_data", "optimal_route", "total_distance",
            "route_error", "route_speed", "search_results", "multiselect_locations",
        ]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()


# ===========================================================================
# หน้าหลัก
# ===========================================================================

def render_main() -> None:

    st.markdown(
        "<div style='"
        "background:linear-gradient(135deg,#0D47A1 0%,#1565C0 50%,#00695C 100%);"
        "padding:28px 32px;border-radius:16px;margin-bottom:24px;text-align:center;"
        "box-shadow:0 4px 20px rgba(0,0,0,0.35)'>"
        "<h1 style='color:white;margin:0 0 8px;font-size:2em'>"
        "🗺️ ระบบแนะนำเส้นทางท่องเที่ยว</h1>"
        "<p style='color:#B3E5FC;margin:0;font-size:1em'>"
        "วางแผนเส้นทางที่ดีที่สุดด้วย <strong>TSP Nearest Neighbour</strong> "
        "+ Merge Sort &amp; Sequential Search</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    gm        = get_graph_manager()
    locations = gm.get_locations()

    # -----------------------------------------------------------------------
    # ส่วน A: รายชื่อสถานที่ (Merge Sort)
    # -----------------------------------------------------------------------
    st.header("📍 สถานที่ท่องเที่ยว")

    if not locations:
        st.info("ยังไม่มีสถานที่ท่องเที่ยว กรุณาเพิ่มสถานที่จากแถบด้านข้าง")
    else:
        sorted_locs = merge_sort(locations)
        n           = len(sorted_locs)

        col_list, col_stats = st.columns([3, 1])
        with col_list:
            df = pd.DataFrame(
                {"ลำดับ": range(1, n + 1), "สถานที่": sorted_locs}
            ).set_index("ลำดับ")
            st.dataframe(df, use_container_width=True, height=min(35 * n + 38, 380))
        with col_stats:
            st.markdown("**สถิติ**")
            st.metric("สถานที่", n)
            st.metric("เส้นทาง", len(gm.get_paths()))

    st.markdown("---")

    # -----------------------------------------------------------------------
    # ส่วน B: รายการเส้นทาง
    # -----------------------------------------------------------------------
    with st.expander("🛣️ เส้นทางการเดินทางทั้งหมด", expanded=False):
        paths = gm.get_paths()
        if paths:
            df_paths = pd.DataFrame(paths).rename(columns={
                "from": "จาก", "to": "ไปยัง", "distance": "ระยะทาง (กม.)",
            })
            st.dataframe(df_paths, use_container_width=True, hide_index=True)
        else:
            st.info("ยังไม่มีเส้นทางการเดินทาง")

    # -----------------------------------------------------------------------
    # ส่วน B2: สถิติกราฟ
    # -----------------------------------------------------------------------
    with st.expander("📊 สถิติและการวิเคราะห์กราฟ", expanded=False):
        g = gm.graph
        if g.number_of_nodes() == 0:
            st.info("เพิ่มสถานที่เพื่อดูสถิติกราฟ")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("สถานที่ (Nodes)", g.number_of_nodes())
            c2.metric("เส้นทาง (Edges)", g.number_of_edges())
            c3.metric("ความหนาแน่น", f"{nx.density(g):.1%}")
            is_conn = nx.is_connected(g)
            c4.metric("เชื่อมต่อทั้งหมด", "✅ ใช่" if is_conn else "❌ ไม่ใช่")

            st.markdown("")
            degrees        = dict(g.degree())
            most_connected = max(degrees, key=degrees.get)
            avg_degree     = sum(degrees.values()) / len(degrees)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(
                    f"**🏆 เชื่อมต่อมากสุด:** {most_connected} "
                    f"({degrees[most_connected]} เส้นทาง)"
                )
                st.markdown(f"**📐 Degree เฉลี่ย:** {avg_degree:.1f}")
                if not is_conn:
                    comps = nx.number_connected_components(g)
                    st.warning(f"⚠️ กราฟแยกเป็น {comps} กลุ่ม — node ที่ยังไม่มีเส้นทางจะแสดงแถวล่างในกราฟ")
            with col_b:
                df_deg = pd.DataFrame(
                    [{"สถานที่": k, "จำนวนเส้นทาง": v}
                     for k, v in sorted(degrees.items(), key=lambda x: -x[1])]
                )
                st.dataframe(df_deg, use_container_width=True, hide_index=True, height=200)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # ส่วน C: คำนวณเส้นทางที่ดีที่สุด
    # -----------------------------------------------------------------------
    st.header("🚗 คำนวณเส้นทางที่ดีที่สุด")
    st.markdown(
        "เลือกสถานที่อย่างน้อย **2 แห่ง** ที่ต้องการเยี่ยมชม "
        "ระบบจะคำนวณเส้นทางที่สั้นที่สุดด้วยอัลกอริทึม **Nearest Neighbour TSP**"
    )

    if len(locations) < 2:
        st.warning("⚠️ กรุณาเพิ่มสถานที่อย่างน้อย 2 แห่งและเส้นทางเชื่อมต่อ")
    else:
        selected = st.multiselect(
            "เลือกสถานที่ที่ต้องการเยี่ยมชม:",
            options=merge_sort(locations),
            key="multiselect_locations",
        )

        start_node = None
        speed      = st.session_state.route_speed

        if len(selected) >= 2:
            opt_col, spd_col = st.columns(2)
            with opt_col:
                start_node = st.selectbox(
                    "🟢 เลือกจุดเริ่มต้น:", options=selected, key="select_start_node"
                )
            with spd_col:
                speed = st.number_input(
                    "🚗 ความเร็วเฉลี่ย (กม./ชม.)",
                    min_value=1.0, max_value=300.0,
                    value=st.session_state.route_speed,
                    step=5.0, key="input_speed",
                    help="ใช้ประมาณเวลาเดินทาง",
                )

        compute_col, _ = st.columns([1, 3])
        with compute_col:
            compute_btn = st.button(
                "🧭 คำนวณเส้นทางที่ดีที่สุด",
                use_container_width=True,
                disabled=(len(selected) < 2),
                key="btn_compute_route",
            )

        if compute_btn and start_node is not None:
            ordered = [start_node] + [s for s in selected if s != start_node]
            try:
                route, total_dist = tsp_nearest_neighbor(gm.graph, ordered)
                st.session_state.optimal_route  = route
                st.session_state.total_distance = total_dist
                st.session_state.route_error    = None
                st.session_state.route_speed    = speed
                st.balloons()
            except ValueError as e:
                st.session_state.route_error    = str(e)
                st.session_state.optimal_route  = None
                st.session_state.total_distance = None

        if st.session_state.route_error:
            st.error(f"❌ {st.session_state.route_error}")

        if st.session_state.optimal_route:
            route = st.session_state.optimal_route
            dist  = st.session_state.total_distance
            spd   = st.session_state.route_speed

            st.success("✅ คำนวณเส้นทางที่ดีที่สุดเรียบร้อยแล้ว!")

            st.markdown("### 🗒️ ลำดับเส้นทางที่แนะนำ")
            for i, loc in enumerate(route):
                if i == 0:
                    st.markdown(f"&nbsp;&nbsp;&nbsp; **🟢 จุดเริ่มต้น:** {loc}")
                elif i == len(route) - 1:
                    st.markdown(f"&nbsp;&nbsp;&nbsp; **🔴 กลับจุดเริ่มต้น:** {loc}")
                else:
                    st.markdown(f"&nbsp;&nbsp;&nbsp; **{i}.** {loc}")

            total_time_min = round(dist / spd * 60) if spd > 0 else None
            m1, m2, m3 = st.columns(3)
            m1.metric("📏 ระยะทางรวม",   f"{dist} กม.")
            m2.metric("🏛️ จำนวนสถานที่", f"{len(route) - 1} แห่ง")
            m3.metric(
                "⏱️ เวลาโดยประมาณ",
                f"~{total_time_min} นาที" if total_time_min else "-",
            )

            st.markdown("### 📋 ตารางรายละเอียดเส้นทาง")
            segments   = []
            cumulative = 0.0
            for i in range(len(route) - 1):
                src, dst = route[i], route[i + 1]
                try:
                    seg_dist = nx.shortest_path_length(gm.graph, src, dst, weight="weight")
                except Exception:
                    seg_dist = 0.0
                cumulative += seg_dist
                seg_time = round(seg_dist / spd * 60, 1) if spd > 0 else None
                segments.append({
                    "ขาที่":         i + 1,
                    "จาก":           src,
                    "ไปยัง":         dst,
                    "ระยะทาง (กม.)": round(seg_dist, 2),
                    "สะสม (กม.)":    round(cumulative, 2),
                    "เวลา (นาที)":   f"~{seg_time}" if seg_time is not None else "-",
                })
            st.dataframe(pd.DataFrame(segments), use_container_width=True, hide_index=True)

            with st.expander("📤 สรุปเส้นทาง (คัดลอกได้)", expanded=False):
                lines = ["🗺️ เส้นทางท่องเที่ยวที่แนะนำ", "─" * 36]
                for i, loc in enumerate(route):
                    if i == 0:
                        lines.append(f"🟢 เริ่มต้น  : {loc}")
                    elif i == len(route) - 1:
                        lines.append(f"🔴 กลับ     : {loc}")
                    else:
                        lines.append(f"  {i:2d}. {loc}")
                lines.append("─" * 36)
                lines.append(f"📏 ระยะทางรวม    : {dist} กม.")
                if total_time_min:
                    lines.append(
                        f"⏱️ เวลาโดยประมาณ : ~{total_time_min} นาที "
                        f"(@ {int(spd)} กม./ชม.)"
                    )
                lines.append(f"🏛️ จำนวนสถานที่   : {len(route) - 1} แห่ง")
                st.text_area("", value="\n".join(lines), height=240, label_visibility="collapsed")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # ส่วน D: กราฟแสดงผล
    # -----------------------------------------------------------------------
    st.header("🖼️ กราฟแสดงสถานที่และเส้นทาง")

    if locations:
        # รวบรวม search matches ที่ยังอยู่ในกราฟ
        search_matches = None
        sr = st.session_state.search_results
        if sr and sr.get("matches"):
            search_matches = [m for m in sr["matches"] if m in gm.graph.nodes()]

        fig = draw_graph(
            gm.graph,
            optimal_route  = st.session_state.optimal_route,
            search_matches = search_matches,
        )
        st.pyplot(fig, use_container_width=True)

        captions = []
        if search_matches:
            captions.append("🔴 สีแดง = ผลการค้นหา")
        if st.session_state.optimal_route:
            captions += [
                "🟢 สีเขียว = จุดเริ่มต้น/สิ้นสุด",
                "🟠 สีส้ม = สถานที่ในเส้นทาง",
                "🔴 เส้นสีแดง = เส้นทางที่ดีที่สุด",
            ]
        if captions:
            st.caption("  |  ".join(captions))
    else:
        st.info("กรุณาเพิ่มสถานที่และเส้นทางเพื่อแสดงกราฟ")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#90A4AE;font-size:12px;padding:8px'>"
        "ระบบแนะนำเส้นทางท่องเที่ยว v2 · Python · NetworkX · Matplotlib · Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
