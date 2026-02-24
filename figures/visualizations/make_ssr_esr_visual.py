import fire

# ==========================================
# 1. STYLE CONFIGURATION
# ==========================================
CONFIG = {
    # -- General Plot Settings --
    "view_az": 45,
    "view_el": 20,
    "width": "0.48\\textwidth",
    "height": "0.4\\textwidth",
    # -- Colors --
    "col_grid_fill": "gray!2",
    "col_grid_lines": "gray!30",
    "col_subspace": "green!60!black",
    "col_mode_true": "blue",
    "col_mode_spur": "red",
    "col_resid_ssr": "violet",
    "col_resid_esr": "orange",
    "col_proj_true": "blue!80",
    "col_proj_spur": "red!80",
    "col_right_angle": "black",
    "caption": (
        r"Geometric illustration of signal-subspace residual (SSR) and "
        r"estimated-subspace residual (ESR) in the case \(m=1\) and \(M=2\), "
        r"where the true signal subspace \(\mathcal{S}\) (green) lies inside "
        r"the truncation subspace \(\mathcal{U}_M = \operatorname{span}\{u_1,u_2\}\). "
        r"True and spurious modes are drawn in blue and red, respectively, together with "
        r"their projections onto \(\mathcal{U}_M\) (dashed). "
        r"(a) SSR corresponds to the in-plane deviation of a mode from \(\mathcal{S}\), "
        r"shown as violet arrows. "
        r"(b) ESR corresponds to the out-of-plane component of a mode, shown as orange arrows. "
        r"In this constructed example, the true mode has small SSR and ESR, "
        r"whereas the spurious mode has larger residual in both senses."
    ),
    # -- Line Widths & Styles --
    "lw_axis": "0.8pt",
    "lw_vector": "1pt",
    "lw_proj": "0.8pt",
    "lw_aux": "0.8pt",  # For dotted lines, arcs
    "style_subspace": "dotted",
    "style_proj": "dashed",
    # -- Arrow Heads --
    "arrow_global": "stealth",  # Sharp technical arrow
    "arrow_ssr": "violet",  # Color for SSR arrow (uses default tip in plot)
    # -- Coordinates (The Geometry) --
    # Subspace S (Green)
    "S_end": "(1.5, 0.23, 0)",
    # True Mode (Blue) - High angle
    "true_tip": "(1, 0.25, 0.1)",
    "true_xy_proj": "(1, 0.25, 0)",
    "true_ssr_start": "(1.0143, 0.1555, 0)",
    # Spurious Mode (Red)
    "spur_tip": "(0.1025660859, 1.025660859, 0.65)",
    "spur_xy_proj": "(0.1025660859, 1.025660859, 0)",
    "spur_ssr_start": "(0.2538, 0.0389, 0)",
}

# ==========================================
# 2. GENERATION LOGIC
# ==========================================


def generate_figure():
    """
    Generates the LaTeX code for the 2-panel leakage figure based on CONFIG.
    """
    c = CONFIG  # Short alias for cleaner f-strings

    # ---------------------------------------------------------
    # Building Blocks (Snippets used in both panels)
    # ---------------------------------------------------------

    # 1. Grid & Basis
    block_grid_basis = f"""
        % --- Grid & Basis ---
        \\fill[{c['col_grid_fill']}] (axis cs:0,0,0) -- (axis cs:1.5,0,0) -- (axis cs:1.5,1.5,0) -- (axis cs:0,1.5,0) -- cycle;
        \\pgfplotsinvokeforeach{{0,0.25,...,1.5}}{{
            \\draw[{c['col_grid_lines']}, thin] (axis cs:#1,0,0) -- (axis cs:#1,1.5,0); 
            \\draw[{c['col_grid_lines']}, thin] (axis cs:0,#1,0) -- (axis cs:1.5,#1,0);
        }}

        % Basis Vectors
        \\addplot3[->, black, line width={c['lw_axis']}] coordinates {{(0,0,0) (1.6,0,0)}}; \\node[anchor=west] at (axis cs:1.6,0,0) {{$u_1$}};
        \\addplot3[->, black, line width={c['lw_axis']}] coordinates {{(0,0,0) (0,1.6,0)}}; \\node[anchor=south] at (axis cs:0,1.6,0) {{$u_2$}};
        \\addplot3[->, black, line width={c['lw_axis']}] coordinates {{(0,0,0) (0,0,0.75)}}; \\node[anchor=south] at (axis cs:0,0,0.73) {{$u_3$}};
        \\fill[black] (axis cs:0,0,0) circle (1pt);
    """

    # 2. Subspace S & Angle Annotation
    block_subspace_angle = f"""
        % --- Subspace S & Angle ---
        \\addplot3[->, {c['style_subspace']}, line width={c['lw_vector']}, draw={c['col_subspace']}] coordinates {{(0,0,0) {c['S_end']}}};
        \\node[anchor=west, text={c['col_subspace']}] at (axis cs:1.5, 0.23, 0) {{$\\mathcal{{S}}$}};

        % Arcsin(eta) Arc
        \\draw[{c['style_subspace']}, black, line width={c['lw_aux']}] (axis cs:1.44, 0, 0) to [bend right=20] (axis cs:1.4, 0.2, 0);
        \\node[anchor=south west, font=\\scriptsize, inner sep=1pt] at (axis cs: 1.45, -0.5, 0.0) {{$\\arcsin(\\eta)$}};
    """

    # 3. Mode Vectors (True/Blue & Spur/Red)
    block_modes = f"""
        % --- Modes ---
        % True Mode (Blue)
        \\addplot3[->, line width={c['lw_vector']}, draw={c['col_mode_true']}] coordinates {{(0,0,0) {c['true_tip']}}};
        \\node[anchor=south east, text={c['col_mode_true']}] at (axis cs:1.3, 0.175, 0.13) {{$\\bm{{\\widehat{{\\phi}}^{{e}}}}_{{ \\text{{true}} }}$}};

        % Spurious Mode (Red)
        \\addplot3[->, line width={c['lw_vector']}, draw={c['col_mode_red'] if 'col_mode_red' in c else c['col_mode_spur']}] coordinates {{(0,0,0) {c['spur_tip']}}};
        \\node[anchor=south east, text={c['col_mode_spur']}] at (axis cs:0.12, 0.66, 0.45) {{$\\bm{{\\widehat{{\\phi}}^{{e}}}}_{{ \\text{{spur}} }}$}};
    """

    # ---------------------------------------------------------
    # Panel Specific Content
    # ---------------------------------------------------------

    # Panel 1: SSR Specifics (Violet arrows + Right Angles)
    content_panel_ssr = f"""
        % --- SSR Vectors (Violet) ---
        % True-SSR (from projection on green line to true mode endpoint)
        \\addplot3[->, line width={c['lw_vector']}, draw={c['col_resid_ssr']}] coordinates {{{c['true_ssr_start']} {c['true_tip']}}};
        \\node[anchor=west, text={c['col_resid_ssr']}, font=\\footnotesize, inner sep=2pt] at (axis cs:1, 0.25, 0.05) {{True-SSR}};

        % Spurious-SSR (from projection on green line to spurious mode endpoint)
        \\addplot3[->, line width={c['lw_vector']}, draw={c['col_resid_ssr']}] coordinates {{{c['spur_ssr_start']} {c['spur_tip']}}};
        \\node[anchor=west, text={c['col_resid_ssr']}, font=\\footnotesize, inner sep=2pt] at (axis cs:0.2, 0.6, 0.4) {{Spurious-SSR}};

        % --- Right Angles (SSR) ---
        % True-SSR Corner
        \\draw[{c['col_right_angle']}, line width=0.4pt]
          (axis cs: 1.0536, 0.1616, 0) --
          (axis cs: 1.0536, 0.1889, 0.0289) --
          (axis cs: 1.0143, 0.1829, 0.0289);

        % Spurious-SSR Corner
        \\draw[{c['col_right_angle']}, line width=0.4pt]
          (axis cs: 0.2931, 0.0450, 0) --
          (axis cs: 0.2886, 0.0843, 0.0285) --
          (axis cs: 0.2493, 0.0782, 0.0285);
    """

    # Panel 2: ESR Specifics (Orange lines + Dashed projections)
    content_panel_esr = f"""
        % --- ESR Components ---
        % True-ESR
        \\addplot3[->, {c['style_proj']}, line width={c['lw_proj']}, draw={c['col_proj_true']}] coordinates {{(0,0,0) {c['true_xy_proj']}}};
        \\addplot3[->, line width={c['lw_vector']}, draw={c['col_resid_esr']}] coordinates {{{c['true_xy_proj']} {c['true_tip']}}};
        \\node[anchor=west, text={c['col_resid_esr']}, font=\\footnotesize] at (axis cs:1, 0.25, 0.05) {{True-ESR}};

        % Spurious-ESR
        \\addplot3[->, {c['style_proj']}, line width={c['lw_proj']}, draw={c['col_proj_spur']}] coordinates {{(0,0,0) {c['spur_xy_proj']}}};
        \\addplot3[-, line width={c['lw_vector']}, draw={c['col_resid_esr']}] coordinates {{{c['spur_xy_proj']} {c['spur_tip']}}};
        \\node[anchor=west, text={c['col_resid_esr']}, font=\\footnotesize] at (axis cs:0.1, 1, 0.4) {{Spurious-ESR}};
    """

    # ---------------------------------------------------------
    # Assemble Full LaTeX
    # ---------------------------------------------------------
    latex_code = f"""
\\begin{{figure*}}[t]
  \\centering
  \\begin{{tikzpicture}}
  \\tikzset{{>={c['arrow_global']}}}
    \\begin{{groupplot}}[
      group style={{
        group size=2 by 1,
        horizontal sep=2cm,
      }},
      %--- Common Axis Options ---
      view={{{c['view_az']}}}{{{c['view_el']}}},
      width={c['width']},
      height={c['height']},
      xmin=0, xmax=1.6,
      ymin=0, ymax=1.6,
      zmin=0, zmax=0.75,
      axis lines=none,
      ticks=none,
      clip=false,
    ]

      %=================================================
      % PANEL 1: SSR FOCUS (Left)
      %=================================================
      \\nextgroupplot[title={{(a) Signal Subspace Residual (SSR)}}]
        {block_grid_basis}
        {block_subspace_angle}
        {block_modes}
        {content_panel_ssr}

      %=================================================
      % PANEL 2: ESR FOCUS (Right)
      %=================================================
      \\nextgroupplot[title={{(b) Estimated Subspace Residual (ESR)}}]
        {block_grid_basis}
        {block_subspace_angle}
        {block_modes}
        {content_panel_esr}

    \\end{{groupplot}}
  \\end{{tikzpicture}}
  \\caption{{{c['caption']}}}
  \\label{{fig:two_panel_residual}}
\\end{{figure*}}
"""
    print(latex_code)


if __name__ == "__main__":
    fire.Fire(generate_figure)
