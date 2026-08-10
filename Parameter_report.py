import os
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PDF_FILENAME = "Immune_Cancer_Simulator_Parameter_Reference.pdf"

# --- Numbered Canvas for Page Headers and Footers ---
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#0f172a"))
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(36, 576, "Immune–Cancer Microenvironment Simulator: Technical Reference Manual")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 570, 756, 570)
        
        # Footer (All Pages)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawString(36, 25, "Confidential — Calibration & Parameter Manual")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(756, 25, page_str)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 35, 756, 35)
        
        self.restoreState()

# --- PDF Builder Function ---
def build_pdf():
    # Landscape orientation for wide parameter tables (792 x 612 pt)
    doc = SimpleDocTemplate(
        PDF_FILENAME,
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#ffffff')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#94a3b8')
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=6,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#334155')
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#0f766e')
    )

    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#1e293b')
    )

    cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#0f172a')
    )

    cell_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#ffffff')
    )

    story = []

    # --- Title Banner ---
    banner_data = [[
        Paragraph("Immune–Cancer Microenvironment Simulator", title_style),
        Paragraph("Status Badges:<br/><b><font color='#166534'>Active</font> | <font color='#991b1b'>Inactive</font> | <font color='#92400e'>Hard-Coded</font> | <font color='#075985'>Compute</font></b>", subtitle_style)
    ], [
        Paragraph("Complete Parameter & Configuration Technical Reference Manual — Version 8.0 (CUDA Engine)", subtitle_style),
        ""
    ]]
    banner_table = Table(banner_data, colWidths=[520, 200])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#0f172a')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,1), (-1,1), 8),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 10))

    # --- Section 1: Executive Overview ---
    story.append(Paragraph("1. Executive Overview & System Architecture", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0f172a'), spaceAfter=8))
    
    overview_text = """
    This document serves as the authoritative parameter reference for the GPU-accelerated 2D agent-based simulator modeling immune–cancer microenvironments in PyTorch.
    The simulator integrates agent movement, continuous chemotaxis, persistent target locking, and bidirectional two-phase combat.
    <br/><br/>
    <b>Key System Guarantees:</b>
    <br/>
    • <b>Temporal Resolution:</b> Internal physics numerical integration executes at <code>dt = 1.0</code> frame/time unit. Camera tracking observation records kinematics every 6 frames (<code>observation_interval = 6</code>), producing an effective observational timestep of <code>Δt_obs = 6.0</code>.
    <br/>
    • <b>Persistent Entity Tracking:</b> Unique GPU tracking IDs are assigned permanently (Immune: <code>1–1000</code>, Cancer: <code>1001–2000</code>). Dead cell IDs are never reused.
    <br/>
    • <b>Domain Auto-Scaling:</b> If explicit width/height are omitted, domain side length auto-scales via <i>Side = √(N / target_density)</i>. For N=2000 and target_density=0.002, <i>Side = 1000.0 μm</i>.
    """
    story.append(Paragraph(overview_text, body_style))
    story.append(Spacer(1, 10))

    # --- Helper Function for Tables ---
    def create_param_table(data_matrix):
        table_data = []
        # Header
        table_data.append([
            Paragraph("Parameter Variable", cell_header),
            Paragraph("Current Value", cell_header),
            Paragraph("Units", cell_header),
            Paragraph("Category", cell_header),
            Paragraph("Status", cell_header),
            Paragraph("Description & Mathematical Effect", cell_header)
        ])
        
        for row in data_matrix:
            var_p = Paragraph(f"<code>{row[0]}</code>", code_style)
            val_p = Paragraph(str(row[1]), cell_bold)
            unit_p = Paragraph(str(row[2]), cell_style)
            cat_p = Paragraph(str(row[3]), cell_style)
            
            status = row[4]
            if status == "Active":
                status_p = Paragraph("<b><font color='#166534'>Active</font></b>", cell_style)
            elif status == "Inactive":
                status_p = Paragraph("<b><font color='#991b1b'>Inactive</font></b>", cell_style)
            elif status == "Hard-Coded":
                status_p = Paragraph("<b><font color='#92400e'>Hard-Coded</font></b>", cell_style)
            else:
                status_p = Paragraph("<b><font color='#075985'>Compute</font></b>", cell_style)

            desc_p = Paragraph(str(row[5]), cell_style)
            table_data.append([var_p, val_p, unit_p, cat_p, status_p, desc_p])

        # Widths total = 720
        t = Table(table_data, colWidths=[120, 80, 50, 60, 50, 360])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        return t

    # --- Section 2: Parameter Reference Tables ---
    story.append(Paragraph("2. Parameter Configuration Tables", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0f172a'), spaceAfter=6))

    # Bucket A
    story.append(Paragraph("BUCKET A — Population & Simulation Configuration", h2_style))
    bucket_a = [
        ["num_immune", "1000", "cells", "Population", "Active", "Initial T-cell count. Increasing increases search density and computational load."],
        ["num_cancer", "1000", "cells", "Population", "Active", "Initial Cancer cell count. Increasing increases spatial crowding."],
        ["width / height", "None (1000.0)", "μm", "Population", "Active", "Domain size. Auto-calculated via √(N / target_density) if None."],
        ["target_density", "0.002", "cells/μm²", "Population", "Active", "Target spatial density used for automatic domain scaling."],
        ["timesteps", "8635", "frames", "Experimental", "Active", "Total internal simulation frames (Frame 0 to 8634)."],
        ["observation_interval", "6", "frames", "Experimental", "Active", "Downsampling sampling interval for CSV kinematics output (Δt_obs = 6.0)."],
        ["mode", "'killing' / 'non-killing'", "string", "Behavioral", "Active", "Toggles active cytotoxic killing vs. suppressed control regime."]
    ]
    story.append(create_param_table(bucket_a))
    story.append(Spacer(1, 10))

    # Bucket B, C, D
    story.append(Paragraph("BUCKETS B, C, D — Phenotype Composition & Motility Speeds", h2_style))
    bucket_bcd = [
        ["scout_prob", "0.30", "ratio", "Biological", "Active", "Fraction of T-cells assigned Scout phenotype (Scout + Msg + Killer = 1.0)."],
        ["messenger_prob", "0.30", "ratio", "Biological", "Active", "Fraction assigned Messenger phenotype (Amplifies recruitment signals)."],
        ["killer_prob", "0.40", "ratio", "Biological", "Active", "Fraction assigned Killer phenotype (Cytotoxic swarmers)."],
        ["IMMUNE_BASE_MEAN", "5.0 (Kill) / 8.5 (Non)", "μm/min", "Motility", "Active", "Mean Gaussian baseline speed distribution for T-cells."],
        ["SPEED_MULTS", "(1.35, 1.00, 0.75)", "multiplier", "Motility", "Active", "Speed scalars: Scout (1.35×), Messenger (1.00×), Killer (0.75×)."],
        ["CAN_SESSILE_SPEED", "(0.5, 1.5)", "μm/min", "Motility", "Active", "Uniform speed range for Sessile Cancer phenotype."],
        ["CAN_EVASIVE_SPEED", "(2.0, 4.0)", "μm/min", "Motility", "Active", "Uniform speed range for Evasive Cancer phenotype."]
    ]
    story.append(create_param_table(bucket_bcd))
    
    story.append(PageBreak()) # --- Page 2 ---

    # Bucket E, F, G, H
    story.append(Paragraph("BUCKETS E, F, G, H — Sensing, Target Locks, & Chemotaxis Signalling", h2_style))
    bucket_efgh = [
        ["SENSING_SCOUT / MSG / KILL", "(24-32), (16-22), (8-14)", "μm", "Sensing", "Active", "Sensing radii for target detection by phenotype."],
        ["RECOG_BASE", "(0.85, 0.60, 0.95)", "probability", "Recognition", "Active", "Base recognition probability (Scout, Messenger, Killer) in killing mode."],
        ["target_lock_timeout", "15", "frames", "Behavioral", "Active", "Steps before persistent target lock expires after losing contact."],
        ["SIGNAL_EMISSION_STRENGTH", "1.00 (Kill) / 0.35 (Non)", "model units", "Signalling", "Active", "Initial chemical signal strength emitted by Scouts."],
        ["CHEMOTAXIS_EPSILON", "2.0", "μm²", "Signalling", "Active", "Softening term in chemotactic weight: Weight = S / (d² + ε)."],
        ["SIGNAL_DECAY_RATE", "0.05", "frame⁻¹", "Signalling", "Active", "Exponential strength decay per step (S_t+1 = S_t × (1 - decay))."],
        ["SIGNAL_LIFETIME_SCOUT/MSG", "25.0 ± 5.0 / 15.0 ± 3.0", "frames", "Signalling", "Active", "Stochastic signal lifetimes (Gaussian sampled, min = 5 frames)."]
    ]
    story.append(create_param_table(bucket_efgh))
    story.append(Spacer(1, 10))

    # Bucket I, J, K
    story.append(Paragraph("BUCKETS I, J, K — Combat, Memory, & Metabolic Energy Dynamics", h2_style))
    bucket_ijk = [
        ["KILL_RADIUS", "2.5", "μm", "Combat", "Active", "Physical threshold distance required for engagement and attack rolls."],
        ["ENGAGE_STEPS_REQUIRED", "4", "frames", "Combat", "Active", "Consecutive contact frames required before primary attack probability executes."],
        ["Killer KILL_RATE", "0.35 (Kill) / 0.0175 (Non)", "probability", "Combat", "Active", "p_kill = (kill_rate + 0.25(1 - e⁻²·exp)) × (0.2 + 0.8 × energy)."],
        ["MEMORY_DECAY / GAIN", "0.005 / 0.30", "step⁻¹ / ratio", "Memory", "Active", "Decay when target lost; Asymptotic recognition gain upon kill."],
        ["ENERGY_DRAIN_MOVE / CHASE", "0.002 / 0.005", "step⁻¹", "Metabolic", "Active", "Metabolic drain rates during standard movement vs. target chasing."],
        ["ENERGY_DRAIN_COMBAT", "0.020", "step⁻¹", "Metabolic", "Inactive", "Defined in codebase but currently unapplied to combat loop."]
    ]
    story.append(create_param_table(bucket_ijk))
    story.append(Spacer(1, 10))

    # Bucket L to S
    story.append(Paragraph("BUCKETS L TO S — Kinematics, Counterattack, & Population Guard", h2_style))
    bucket_l_to_s = [
        ["tau (Scout, Msg, Kill, Can)", "(1-2), (2-3.5), (3.5-5), (2-4)", "frames", "Kinematics", "Active", "Steering persistence time constant: a = (v_desired - v) / τ."],
        ["noise_scale", "Scout(0.2-0.3), Killer(0.01-0.05)", "μm/min²", "Kinematics", "Active", "Random motility noise: eff_noise = noise_scale × (1.5 - 0.5 × energy)."],
        ["ALIGNMENT_RADIUS_KILL/CAN", "12.0 / 10.0", "μm", "Collective", "Active", "Spatial radius for neighbor velocity alignment swarming."],
        ["MIN_SEPARATION", "1.5", "μm", "Physics", "Active", "Cell collision hard sphere radius. Triggers linear repulsive force."],
        ["BOUNDARY_MARGIN", "8.0", "μm", "Physics", "Active", "Distance from arena boundary where wall repelling push force activates."],
        ["Counterattack Base", "0.03 (Sessile) / 0.09 (Evasive)", "probability", "Counterattack", "Active", "p_counter = base × (0.3 + 0.7×E) / (1 + 0.5 × local_immune_count)."],
        ["enable_proliferation / apop", "False / False", "boolean", "Population", "Inactive", "Strictly disabled for experimental tracking calibration."]
    ]
    story.append(create_param_table(bucket_l_to_s))

    story.append(PageBreak()) # --- Page 3 ---

    # --- Section 3 & 4: Hard-Coded Constants ---
    story.append(Paragraph("3. Hard-Coded Constants Recommended for Config Exposure", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0f172a'), spaceAfter=6))
    
    hc_data = [
        ["Hard-Coded Constant Description", "Value", "Location in Code", "Recommended Parameter Variable Name"],
        ["Recognition Noise Standard Dev", "0.08", "_initialize_motion_parameters", "RECOG_NOISE_STD"],
        ["Messenger Signal Amplification Scalar", "1.4 ×", "update_immune_cells", "MSG_AMPLIFY_MULT"],
        ["Signal Direction / Alignment Blend", "0.70 / 0.30", "update_immune_cells", "IMMUNE_SWARM_BLEND"],
        ["Cancer Flee / Alignment Blend", "0.75 / 0.25", "update_cancer_cells", "CANCER_FLEE_BLEND"],
        ["Collision Repulsion Multiplier", "0.5", "resolve_collisions", "REPULSION_FORCE_COEFF"],
        ["Boundary Wall Push Force Scalar", "0.3", "apply_boundary_forces", "BOUNDARY_PUSH_COEFF"],
        ["Cancer Velocity Damping Factor", "0.10", "update_cancer_cells", "CANCER_DAMPING_COEFF"],
        ["Counterattack Engagement Steps", "5 frames", "perform_killing_phase2", "COUNTER_ENGAGE_STEPS"],
        ["Counterattack Pressure Coefficient", "0.5", "perform_killing_phase2", "COUNTER_PRESSURE_COEFF"],
        ["Combat Experience Gain / Max Cap", "+0.10 / 1.0", "perform_killing_phase1", "EXP_GAIN_PER_KILL"]
    ]
    
    hc_table_data = []
    for idx, row in enumerate(hc_data):
        if idx == 0:
            hc_table_data.append([Paragraph(c, cell_header) for c in row])
        else:
            hc_table_data.append([
                Paragraph(row[0], cell_style),
                Paragraph(row[1], cell_bold),
                Paragraph(f"<code>{row[2]}</code>", code_style),
                Paragraph(f"<code>{row[3]}</code>", code_style)
            ])
            
    t_hc = Table(hc_table_data, colWidths=[200, 80, 220, 220])
    t_hc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_hc)
    story.append(Spacer(1, 10))

    # --- Section 5: Calibration Hierarchy ---
    story.append(Paragraph("4. Recommended Calibration Hierarchy Framework", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0f172a'), spaceAfter=6))
    
    tier_data = [
        [Paragraph("Priority Tier", cell_header), Paragraph("Focus Domain", cell_header), Paragraph("Target Calibration Parameters", cell_header)],
        [Paragraph("TIER 1 (Highest)", cell_bold), Paragraph("Single-Cell Kinematics & Motility", cell_style), Paragraph("<code>IMMUNE_BASE_MEAN</code>, <code>SPEED_MULTS</code>, <code>CAN_EVASIVE_SPEED</code>, <code>tau</code>, <code>noise_scale</code>", cell_style)],
        [Paragraph("TIER 2", cell_bold), Paragraph("Sensing & Local Engagement", cell_style), Paragraph("<code>SENSING_SCOUT/KILLER</code>, <code>RECOG_BASE</code>, <code>KILL_RADIUS</code>, <code>KILL_RATE</code>, <code>target_lock_timeout</code>", cell_style)],
        [Paragraph("TIER 3", cell_bold), Paragraph("Chemotaxis & Collective Swarming", cell_style), Paragraph("<code>SIGNAL_EMISSION_STRENGTH</code>, <code>SIGNAL_DECAY_RATE</code>, <code>ALIGNMENT_RADIUS</code>, Swarm Weights", cell_style)],
        [Paragraph("TIER 4", cell_bold), Paragraph("Adaptive Memory & Metabolic State", cell_style), Paragraph("<code>MEMORY_GAIN_FACTOR</code>, <code>MEMORY_DECAY_RATE</code>, <code>ENERGY_DRAIN_CHASE/MOVE</code>", cell_style)],
        [Paragraph("TIER 5 (Fixed)", cell_bold), Paragraph("Experimental Protocol Constants", cell_style), Paragraph("<code>observation_interval (6)</code>, <code>dt (1.0)</code>, <code>num_immune (1000)</code>, <code>num_cancer (1000)</code>", cell_style)],
    ]
    t_tier = Table(tier_data, colWidths=[90, 180, 450])
    t_tier.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_tier)

    story.append(PageBreak()) # --- Page 4 ---

    # --- Section 6: Master Parameter Dashboard ---
    story.append(Paragraph("5. Master Parameter Calibration Dashboard", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0f172a'), spaceAfter=8))

    dash_data = [
        [Paragraph("MOTILITY & KINEMATICS", cell_header), Paragraph("SIGNALLING & COMBAT", cell_header), Paragraph("MEMORY, PHYSICS, & PROTOCOL", cell_header)],
        [
            Paragraph("""
            • <code>IMMUNE_BASE_MEAN</code> = 5.0 / 8.5 μm/min<br/>
            • <code>SPEED_MULTS</code> = (1.35, 1.00, 0.75)<br/>
            • <code>CAN_SESSILE_SPEED</code> = (0.5, 1.5) μm/min<br/>
            • <code>CAN_EVASIVE_SPEED</code> = (2.0, 4.0) μm/min<br/>
            • <code>Scout tau</code> = 1.0 – 2.0 frames<br/>
            • <code>Killer tau</code> = 3.5 – 5.0 frames<br/>
            • <code>Scout noise</code> = 0.20 – 0.30<br/>
            • <code>Killer noise</code> = 0.01 – 0.05
            """, cell_style),
            Paragraph("""
            • <code>SIGNAL_EMISSION_STR</code> = 1.00<br/>
            • <code>SIGNAL_DECAY_RATE</code> = 0.05 frame⁻¹<br/>
            • <code>SIGNAL_SENSING_RAD</code> = 30.0 μm<br/>
            • <code>CHEMOTAXIS_EPSILON</code> = 2.0 μm²<br/>
            • <code>KILL_RADIUS</code> = 2.5 μm<br/>
            • <code>ENGAGE_STEPS_REQUIRED</code> = 4 frames<br/>
            • <code>Killer KILL_RATE</code> = 0.35<br/>
            • <code>Counterattack Base</code> = 0.03 / 0.09
            """, cell_style),
            Paragraph("""
            • <code>MEMORY_GAIN_FACTOR</code> = 0.30<br/>
            • <code>MEMORY_DECAY_RATE</code> = 0.005<br/>
            • <code>ALIGNMENT_RADIUS</code> = 12.0 μm<br/>
            • <code>MIN_SEPARATION</code> = 1.5 μm<br/>
            • <code>BOUNDARY_MARGIN</code> = 8.0 μm<br/>
            • <code>dt</code> = 1.0 | <code>dt_obs</code> = 6.0<br/>
            • <code>timesteps</code> = 8635 (1439 recorded)<br/>
            • <code>N</code> = 2000 (1000 Immune / 1000 Cancer)
            """, cell_style)
        ]
    ]
    t_dash = Table(dash_data, colWidths=[240, 240, 240])
    t_dash.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f766e')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#0f766e')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#f0fdf4')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_dash)
    story.append(Spacer(1, 15))

    # Calibration Notice Box
    notice_data = [[
        Paragraph("<b>IMPORTANT CALIBRATION NOTICE:</b> Parameter values listed in this document reflect the baseline simulator configuration. All biological and motility parameters are subject to Tier 1–4 optimization against experimental live-cell tracking datasets.", ParagraphStyle('Notice', parent=body_style, textColor=colors.HexColor('#78350f')))
    ]]
    t_notice = Table(notice_data, colWidths=[720])
    t_notice.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fffbe2')),
        ('BORDER', (0,0), (-1,-1), 0.5, colors.HexColor('#fde68a')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_notice)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated: {os.path.abspath(PDF_FILENAME)}")

if __name__ == "__main__":
    build_pdf()