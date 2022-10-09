import os
import sys
import geopandas as gpd
from leafmap.common import classify
import leafmap.foliumap as leafmap
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import textwrap as tr
from streamlit.web import cli as stcli
from skimage.color import rgb2hsv


def main(workspace):
    # helper functions
    # color mapping
    def rgb_to_hex(rgb_df):
        rgbs = (255 * rgb_df).astype("int")
        rgbs = [
            (r, g, b)
            for r, g, b in zip(rgbs["embed_I"], rgbs["embed_II"], rgbs["embed_III"])
        ]
        hex_map = lambda x: "#" + "".join(f"{i:02X}" for i in x)
        hexs = [hex_map(rgb) for rgb in rgbs]
        return hexs

    # define paths
    workspace_root = os.path.split(workspace)[0]
    workdir = os.path.join(workspace_root, "results", "FeatureEDA")
    save_dir = workdir
    os.makedirs(save_dir, exist_ok=True)

    # read data
    f_vals = pd.read_csv(os.path.join(workdir, "feature_vals.csv"))
    f_vals_ztrans = (f_vals - f_vals.mean()) / f_vals.std()
    f_sets = pd.read_csv(os.path.join(workdir, "feature_sets.csv"))

    def load_data():
        if hasattr(st.session_state, "usr_fset"):
            f_set_name = st.session_state.usr_fset
        else:
            f_set_name = f_sets.columns[0]
        f_set_feats = list(f_sets.loc[:, f_set_name].dropna())
        em_spec = gpd.read_file(os.path.join(workdir, f"embeddings_{f_set_name}.shp"))
        em_rgb = em_spec.drop(columns="geometry").apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        return f_set_name, f_set_feats, em_spec, em_rgb

    f_set_name, f_set_feats, em_spec, em_rgb = load_data()

    # dashboard elements
    # map
    m = leafmap.Map(
        center=(40, -100),
        zoom=4,
        draw_control=False,
        measure_control=False,
    )
    m.add_basemap("SATELLITE")
    gdf, _ = classify(
        data=em_spec.reset_index(),
        column="index",
        colors=rgb_to_hex(em_rgb),
        k=len(em_spec),
    )
    style = lambda feat: {
        "color": "#000000",
        "weight": 0.5,
        "opacity": 1,
        "fillOpacity": 1.0,
        "fillColor": feat["properties"]["color"],
    }
    m.add_gdf(gdf, layer_name="embedding", info_mode="on_click", style_function=style)

    # scatterplot
    @st.cache
    def create_scatterplot():
        fig = px.scatter_3d(
            em_spec,
            x="embed_I",
            y="embed_II",
            z="embed_III",
            color_continuous_scale=rgb_to_hex(em_rgb),
            color=em_spec.index,
            hover_data={"embed_I": ":.1f", "embed_II": ":.1f", "embed_III": ":.1f"},
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(
            scene=dict(
                xaxis_title="",
                yaxis_title="",
                zaxis_title="",
                # xaxis=dict(linecolor="#000000", linewidth=2,showbackground=False),
            )
        )
        return fig

    # boxplot
    @st.cache
    def create_boxplot(obj_df, col_ls, canvas_setting):
        # reformat variables labels to minimise ovlp
        obj_df["variable"] = [
            f"</br>{'</br>'.join(tr.wrap(x, width=12))}" for x in obj_df["variable"]
        ]
        # create figure
        fig = go.Figure(
            data=[
                go.Box(
                    x=obj_df[obj_df["group"] == group]["variable"],
                    y=obj_df[obj_df["group"] == group]["value"],
                    marker_color=col_ls[group],
                    hoverinfo="none",
                )
                for group in obj_df["group"].unique()
            ]
        )
        # modify figure
        if canvas_setting == "lightmode":
            fontcolor = "rgb(0,0,0)"
            bgcolor = "rgb(245,245,245)"
        elif canvas_setting == "darkmode":
            fontcolor = "rgb(245,245,245)"
            bgcolor = "rgb(0,0,0)"
        fig.update_layout(
            xaxis=dict(
                color=fontcolor,
                title_font_color=fontcolor,
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                color=fontcolor,
                title_font_color=fontcolor,
                gridcolor="white",
                title="standardised values",
            ),
            paper_bgcolor=bgcolor,
            plot_bgcolor=bgcolor,
            showlegend=False,
            boxmode="group",
        )
        return fig

    # descriptive stats on original features
    # part I - color selection based on colors currently present
    @st.cache(show_spinner=False)
    def create_cmap():
        # colors sorting in hsv space
        rgbs = [
            (r, g, b)
            for r, g, b in zip(
                em_rgb["embed_I"], em_rgb["embed_II"], em_rgb["embed_III"]
            )
        ]
        rgbs.sort(key=lambda rgb: rgb2hsv(np.array(rgb))[0])
        # smoothen discrete colors
        mv_avg = lambda x, w: np.convolve(x, np.ones(w), "valid") / w
        kernel_s = int(len(rgbs) * 0.01)
        rgbs = np.apply_along_axis(mv_avg, 0, np.array(rgbs), kernel_s)
        rgbs = [(r, g, b) for r, g, b in rgbs]
        # create colormap
        fig_cmp, ax = plt.subplots(figsize=(15, 0.5), frameon=False)
        ax.bar(np.arange(len(rgbs)), height=1, width=1, color=rgbs)
        fig_cmp.axes[0].margins(x=0)
        fig_cmp.axes[0].set_axis_off()
        fig_cmp.savefig(
            os.path.join(save_dir, f"colors_{f_set_name}.png"),
            pad_inches=0,
            bbox_inches="tight",
        )
        return rgbs

    # part II - get corresponding objects
    # eval user selection (see callbacks)
    def get_objs(spec_color, neighbours, group=0):
        slct_col = rgbs[int(len(rgbs) * spec_color)]
        diffs = np.array(em_rgb) - np.array(len(em_rgb) * [slct_col])
        diffs = np.abs(diffs).sum(axis=1)
        slct_objs = em_rgb.iloc[np.argsort(diffs)[:neighbours], :]
        slct_objs = pd.melt(f_vals_ztrans.loc[slct_objs.index, f_set_feats])
        slct_objs["group"] = group
        return slct_col, slct_objs

    # app layout
    st.set_page_config(layout="wide")
    # sidebar
    st.sidebar.title("Exploratory Data Analysis - OBIA Feature EDA")
    st.sidebar.info(
        """
    This dashboard is supposed to support feature analyses by
    providing some interactive exploration tools. Please note that
    the current version is only a prototype.
    """,
        icon="ℹ",
    )
    st.sidebar.title("Further information")
    st.sidebar.info(
        """
        [GitHub repository](https://github.com/fkroeber/obia_exploratory)
        """
    )
    # main page
    st.header("Feature set to analyse")
    usr_fset = st.radio(
        "Available feature sets",
        list(f_sets.columns),
        horizontal=True,
        label_visibility="collapsed",
        key="usr_fset",
        on_change=load_data,
    )
    st.text("")
    con1 = st.container()
    con2 = st.container()
    with con1:
        col1, _, col2 = st.columns([8, 1, 8])
        with col1:
            st.header("Map visualisation")
            st.caption(
                """
            This visualisation is basically the same as the one in ecognition.
            It allows to explore the results from a spatial perspective.
            """
            )
            st.text("")
            m.to_streamlit(height=400)
        with col2:
            st.header("Embedded Feature Space")
            st.caption(
                """
            This visualisation puts the focus on the organisation of the feature space.
            However, be careful when interpreting the structural composition in terms of
            clustering effects and distances.  [How (not) to misread UMAP](https://pair-code.github.io/understanding-umap/)     
            """
            )
            scatter = create_scatterplot()
            st.plotly_chart(scatter, use_container_width=True)
    with con2:
        st.header("Underlying original features")
        st.caption(
            """
            This visualisation shows the connection between embedded & original values.
            For objects that are in relative proximity to each other in the embedded feature space 
            (i.e. have the same colour), the range of values over the original features is shown.
            This allows to further evaluate the degree and nature of commonality of similarly presented objects.
            """
        )
        with st.expander("Simple configuration (single group)"):
            usr_n_obj = st.slider(
                """
                Use this slider to select the number of similar objects to be included in the analyses.
                More neighbours necessarily lead to the inclusion of less similar ones.
                By default there is a reasonable default set in a way that a cluster of similar objects is put together.
                """,
                10,
                50,
                30,
                1,
            )
            usr_col_I = st.slider(
                "Use this slider to select the color for the group of objects your interested in.",
                0.0,
                1.0,
                0.5,
                0.001,
            )
            rgbs = create_cmap()
            st.image(
                image.imread(os.path.join(save_dir, f"colors_{f_set_name}.png")),
                use_column_width=True,
            )
        with st.expander("Advanced configuration (multi-group)"):
            usr_group_comp = st.checkbox(
                "Make comparison against a second group of objects"
            )
            usr_col_II = st.slider(
                "Use this slider to select the color for the group of objects you want to compare the first group with.",
                0.0,
                1.0,
                0.5,
                0.001,
            )
            rgbs = create_cmap()
            st.image(
                image.imread(os.path.join(save_dir, f"colors_{f_set_name}.png")),
                use_column_width=True,
            )
        with st.expander("Other plotting settings"):
            usr_box_canvas = st.radio(
                "Plotting canvas",
                ["lightmode", "darkmode"],
                horizontal=True,
                label_visibility="collapsed",
            )
        # get user selected objects & cols
        if usr_group_comp:
            slct_cols_I, slct_objs_I = get_objs(usr_col_I, usr_n_obj, 0)
            slct_cols_II, slct_objs_II = get_objs(usr_col_II, usr_n_obj, 1)
            slct_objs = pd.concat((slct_objs_I, slct_objs_II))
            slct_cols = [slct_cols_I, slct_cols_II]
        else:
            slct_cols, slct_objs = get_objs(usr_col_I, usr_n_obj, 0)
            slct_cols = [slct_cols]
        slct_cols = (255 * np.array(slct_cols)).astype("int32")
        slct_cols = [f"rgb({rgb[0]},{rgb[1]},{rgb[2]})" for rgb in slct_cols]
        # show boxplot
        box = create_boxplot(slct_objs, slct_cols, usr_box_canvas)
        st.plotly_chart(box, use_container_width=True)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main(sys.argv[1])
    else:
        sys.argv = ["streamlit", "run", sys.argv[0], sys.argv[1]]
        sys.exit(stcli.main())
