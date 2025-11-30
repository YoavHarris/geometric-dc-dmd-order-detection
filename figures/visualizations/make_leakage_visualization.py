import fire

# ==========================================
# 1. STYLE CONFIGURATION
# ==========================================
CONFIG = {
    # -- General Plot Settings --
    "view_az": 45,
    "view_el": 20,
    "width": "0.48\\textwidth",
    "height": "0.5\\textwidth",
    # -- Colors --
    "col_grid_fill": "gray!2",
    "col_grid_lines": "gray!30",
    "col_subspace": "green!60!black",
    "col_mode_true": "blue",
    "col_mode_spur": "red",
    "col_leak_ssl": "violet",
    "col_leak_esl": "orange",
    "col_proj_true": "blue!80",
    "col_proj_spur": "red!80",
    "col_right_angle": "black!60",
    "caption": (
        r"Geometric illustration of signal-subspace leakage (SSL) and "
        r"estimated-subspace leakage (ESL) in the case \(m=1\) and \(M=2\), "
        r"where the true signal subspace \(\mathcal{S}\) (green) lies inside "
        r"the truncation subspace \(\mathcal{U}_M = \operatorname{span}\{u_1,u_2\}\). "
        r"True and spurious modes are drawn in blue and red, respectively, together with "
        r"their projections onto \(\mathcal{U}_M\) (dashed). "
        r"(a) SSL corresponds to the in-plane deviation of a mode from \(\mathcal{S}\), "
        r"shown as violet arrows. "
        r"(b) ESL corresponds to the out-of-plane component of a mode, shown as orange arrows. "
        r"In this constructed example, the true mode has small SSL and ESL, "
        r"whereas the spurious mode has larger leakage in both senses."
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
    "arrow_ssl": "violet",  # Color for SSL arrow (uses default tip in plot)
    # -- Coordinates (The Geometry) --
    # Subspace S (Green)
    "S_end": "(1.5, 0.15, 0)",
    # True Mode (Blue) - High angle
    "true_tip": "(1, 0.25, 0.2)",
    "true_xy_proj": "(1, 0.25, 0)",
    "true_ssl_start": "(1, 0.1, 0)",
    # Spurious Mode (Red)
    "spur_tip": "(0.1, 1, 0.8)",
    "spur_xy_proj": "(0.1, 1, 0)",
    "spur_ssl_start": "(0.15, 0.015, 0)",
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
        \\addplot3[->, black, line width={c['lw_axis']}] coordinates {{(0,0,0) (0,0,1.5)}}; \\node[anchor=south] at (axis cs:0,0,1.5) {{$u_3$}};
        \\fill[black] (axis cs:0,0,0) circle (1pt);
    """

    # 2. Subspace S & Angle Annotation
    block_subspace_angle = f"""
        % --- Subspace S & Angle ---
        \\addplot3[->, {c['style_subspace']}, line width={c['lw_vector']}, draw={c['col_subspace']}] coordinates {{(0,0,0) {c['S_end']}}};
        \\node[anchor=west, text={c['col_subspace']}] at (axis cs:1.5, 0.15, 0) {{$\\mathcal{{S}}$}};

        % Arcsin(eta) Arc
        \\draw[{c['style_subspace']}, black, line width={c['lw_aux']}] (axis cs:1.44, 0, 0) to [bend right=15] (axis cs:1.43, 0.15, 0);
        \\node[anchor=south west, font=\\scriptsize, inner sep=1pt] at (axis cs: 1.4, -0.45, 0) {{$\\arcsin(\\eta)$}};
    """

    # 3. Mode Vectors (True/Blue & Spur/Red)
    block_modes = f"""
        % --- Modes ---
        % True Mode (Blue)
        \\addplot3[->, line width={c['lw_vector']}, draw={c['col_mode_true']}] coordinates {{(0,0,0) {c['true_tip']}}};
        \\node[anchor=south east, text={c['col_mode_true']}] at (axis cs:0.85, 0.2, 0.15) {{$\\bm{{\\widehat{{\\phi}}^{{e}}}}_{{ \\text{{true}} }}$}};

        % Spurious Mode (Red)
        \\addplot3[->, line width={c['lw_vector']}, draw={c['col_mode_red'] if 'col_mode_red' in c else c['col_mode_spur']}] coordinates {{(0,0,0) {c['spur_tip']}}};
        \\node[anchor=south east, text={c['col_mode_spur']}] at (axis cs:0.1, 0.6, 0.45) {{$\\bm{{\\widehat{{\\phi}}^{{e}}}}_{{ \\text{{spur}} }}$}};
    """

    # ---------------------------------------------------------
    # Panel Specific Content
    # ---------------------------------------------------------

    # Panel 1: SSL Specifics (Violet arrows + Right Angles)
    content_panel_ssl = f"""
        % --- Right Angles (SSL) ---
        % True-SSL Corner
        \\draw[{c['col_right_angle']}, thin]
          (axis cs: 1.08, 0.108, 0) --      
          (axis cs: 1.08, 0.15, 0.05) --    
          (axis cs: 1.0, 0.14, 0.05);       

        % Spurious-SSL Corner
        \\draw[{c['col_right_angle']}, thin]
          (axis cs: 0.23, 0.023, 0) --      
          (axis cs: 0.22, 0.12, 0.08) --    
          (axis cs: 0.14, 0.11, 0.08);      

        % --- SSL Vectors (Violet) ---
        % True-SSL
        \\addplot3[->, line width={c['lw_vector']}, draw={c['col_leak_ssl']}] coordinates {{{c['true_ssl_start']} {c['true_tip']}}};
        \\node[anchor=west, text={c['col_leak_ssl']}, font=\\footnotesize, inner sep=2pt] at (axis cs:1.05, 0.15, 0.1) {{True-SSL}};

        % Spurious-SSL
        \\addplot3[->, line width={c['lw_vector']}, draw={c['col_leak_ssl']}] coordinates {{{c['spur_ssl_start']} {c['spur_tip']}}};
        \\node[anchor=west, text={c['col_leak_ssl']}, font=\\footnotesize, inner sep=2pt] at (axis cs:0.1, 0.6, 0.4) {{Spurious-SSL}};
    """

    # Panel 2: ESL Specifics (Orange lines + Dashed projections)
    content_panel_esl = f"""
        % --- ESL Components ---
        % True-ESL
        \\addplot3[->, {c['style_proj']}, line width={c['lw_proj']}, draw={c['col_proj_true']}] coordinates {{(0,0,0) {c['true_xy_proj']}}};
        \\addplot3[-, line width={c['lw_vector']}, draw={c['col_leak_esl']}] coordinates {{{c['true_xy_proj']} {c['true_tip']}}};
        \\node[anchor=west, text={c['col_leak_esl']}, font=\\footnotesize] at (axis cs:1, 0.25, 0.1) {{True-ESL}};

        % Spurious-ESL
        \\addplot3[->, {c['style_proj']}, line width={c['lw_proj']}, draw={c['col_proj_spur']}] coordinates {{(0,0,0) {c['spur_xy_proj']}}};
        \\addplot3[-, line width={c['lw_vector']}, draw={c['col_leak_esl']}] coordinates {{{c['spur_xy_proj']} {c['spur_tip']}}};
        \\node[anchor=west, text={c['col_leak_esl']}, font=\\footnotesize] at (axis cs:0.1, 1, 0.4) {{Spurious-ESL}};
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
      zmin=0, zmax=1.5,
      axis lines=none,
      ticks=none,
      clip=false,
    ]

      %=================================================
      % PANEL 1: SSL FOCUS (Left)
      %=================================================
      \\nextgroupplot[title={{(a) Signal Subspace Leakage (SSL)}}]
        {block_grid_basis}
        {block_subspace_angle}
        {block_modes}
        {content_panel_ssl}

      %=================================================
      % PANEL 2: ESL FOCUS (Right)
      %=================================================
      \\nextgroupplot[title={{(b) Estimated Subspace Leakage (ESL)}}]
        {block_grid_basis}
        {block_subspace_angle}
        {block_modes}
        {content_panel_esl}

    \\end{{groupplot}}
  \\end{{tikzpicture}}
  \\caption{{{c['caption']}}}
  \\label{{fig:two_panel_leakage}}
\\end{{figure*}}
"""
    print(latex_code)


if __name__ == "__main__":
    fire.Fire(generate_figure)
