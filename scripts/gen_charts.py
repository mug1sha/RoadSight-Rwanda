"""Generate all chart JSON for the premium RoadSight Rwanda dashboard."""
import pandas as pd, numpy as np, json
import plotly.graph_objects as go

df = pd.read_csv('data/processed/accidents_clean.csv')
df['crash_datetime'] = pd.to_datetime(df['crash_datetime'])

SEV_ORDER  = ['Damage only', 'Minor injury', 'Serious/grievous', 'Fatal']
SEV_COLORS = ['#2ed573', '#ffa502', '#ff6b35', '#ff4757']
DARK = 'plotly_dark'
BG   = 'rgba(0,0,0,0)'
GRID = '#1e2a3a'
TICK = '#8899aa'
TITLE_FONT = dict(size=16, color='#e0e0e0')
MARGIN = dict(t=55, b=45, l=55, r=20)

charts = {}

# ── 1. Severity donut ──────────────────────────────────────────────────────────
sev_counts = df['severity'].value_counts().reindex(SEV_ORDER)
fig1 = go.Figure(go.Pie(
    labels=SEV_ORDER, values=sev_counts.values, hole=0.58,
    marker_colors=SEV_COLORS,
    textinfo='label+percent',
    textfont=dict(size=12, color='#e0e0e0'),
    hovertemplate='<b>%{label}</b><br>%{value} accidents<br>%{percent}<extra></extra>',
))
fig1.update_layout(
    template=DARK, paper_bgcolor=BG, plot_bgcolor=BG,
    title=dict(text='Severity Distribution', font=TITLE_FONT),
    legend=dict(orientation='h', y=-0.18, font=dict(color='#aaa')),
    height=380, margin=dict(t=55, b=70, l=20, r=20)
)
charts['donut'] = fig1.to_json()

# ── 2. 3D Surface hour × day ──────────────────────────────────────────────────
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
surf = np.zeros((7, 24))
for _, row in df.iterrows():
    surf[int(row['crash_dow'])][int(row['crash_hour'])] += 1

fig2 = go.Figure(go.Surface(
    z=surf, x=list(range(24)), y=dow_names,
    colorscale=[
        [0.0,  '#050a14'],
        [0.25, '#003566'],
        [0.55, '#0070cc'],
        [0.80, '#00d4ff'],
        [1.0,  '#ff4757'],
    ],
    contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True))),
    hovertemplate='Hour: %{x}:00<br>Day: %{y}<br>Accidents: %{z:.0f}<extra></extra>',
    showscale=True,
    colorbar=dict(tickfont=dict(color='#aaa'), title=dict(text='Count', font=dict(color='#aaa'))),
))
fig2.update_layout(
    template=DARK, paper_bgcolor=BG,
    title=dict(text='3D Accident Surface — Hour × Day of Week', font=TITLE_FONT),
    scene=dict(
        xaxis=dict(title=dict(text='Hour of Day', font=dict(color='#00d4ff', size=11)),
                   tickfont=dict(color=TICK), gridcolor='#1e2a3a', backgroundcolor=BG),
        yaxis=dict(title=dict(text=''), tickfont=dict(color=TICK),
                   gridcolor='#1e2a3a', backgroundcolor=BG),
        zaxis=dict(title=dict(text='Accidents', font=dict(color='#00d4ff', size=11)),
                   tickfont=dict(color=TICK), gridcolor='#1e2a3a', backgroundcolor=BG),
        camera=dict(eye=dict(x=1.7, y=-1.7, z=1.1)),
        bgcolor=BG,
    ),
    height=500, margin=dict(t=55, b=10, l=10, r=10)
)
charts['surface3d'] = fig2.to_json()

# ── 3. Hourly stacked bar ─────────────────────────────────────────────────────
hourly = (df.groupby(['crash_hour', 'severity'])
            .size().unstack(fill_value=0)
            .reindex(columns=SEV_ORDER, fill_value=0))
fig3 = go.Figure()
for sev, color in zip(SEV_ORDER, SEV_COLORS):
    fig3.add_trace(go.Bar(
        x=list(range(24)), y=hourly[sev].values, name=sev,
        marker_color=color,
        hovertemplate=f'<b>{sev}</b><br>Hour %{{x}}:00 — %{{y}} accidents<extra></extra>',
    ))
fig3.update_layout(
    template=DARK, barmode='stack', paper_bgcolor=BG, plot_bgcolor=BG,
    title=dict(text='Accidents by Hour of Day', font=TITLE_FONT),
    xaxis=dict(title='Hour', tickmode='linear', dtick=2,
               tickfont=dict(color=TICK), gridcolor=GRID, title_font=dict(color=TICK)),
    yaxis=dict(title='Count', tickfont=dict(color=TICK), gridcolor=GRID, title_font=dict(color=TICK)),
    legend=dict(font=dict(color='#aaa'), bgcolor=BG),
    height=360, margin=MARGIN
)
charts['hourly'] = fig3.to_json()

# ── 4. Province grouped bar ───────────────────────────────────────────────────
prov_sev = (df.groupby(['province', 'severity'])
              .size().unstack(fill_value=0)
              .reindex(columns=SEV_ORDER, fill_value=0))
fig4 = go.Figure()
for sev, color in zip(SEV_ORDER, SEV_COLORS):
    fig4.add_trace(go.Bar(
        x=prov_sev.index.tolist(), y=prov_sev[sev].values,
        name=sev, marker_color=color,
    ))
fig4.update_layout(
    template=DARK, barmode='group', paper_bgcolor=BG, plot_bgcolor=BG,
    title=dict(text='Severity by Province', font=TITLE_FONT),
    xaxis=dict(tickfont=dict(color='#e0e0e0'), gridcolor=GRID),
    yaxis=dict(title='Count', tickfont=dict(color=TICK), gridcolor=GRID, title_font=dict(color=TICK)),
    legend=dict(font=dict(color='#aaa'), bgcolor=BG),
    height=380, margin=MARGIN
)
charts['province'] = fig4.to_json()

# ── 5. Monthly fatal area chart ───────────────────────────────────────────────
monthly = (df[df['severity'] == 'Fatal']
           .groupby(['crash_year', 'crash_month']).size()
           .reset_index(name='count'))
monthly['date'] = pd.to_datetime(
    monthly['crash_year'].astype(str) + '-' + monthly['crash_month'].astype(str).str.zfill(2) + '-01'
)
monthly = monthly.sort_values('date')

fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=monthly['date'], y=monthly['count'], mode='lines+markers',
    fill='tozeroy', fillcolor='rgba(255,71,87,0.12)',
    line=dict(color='#ff4757', width=2.5),
    marker=dict(color='#ff4757', size=6, line=dict(color='#fff', width=1)),
    hovertemplate='<b>%{x|%b %Y}</b><br>%{y} fatal accidents<extra></extra>',
))
fig5.update_layout(
    template=DARK, paper_bgcolor=BG, plot_bgcolor=BG,
    title=dict(text='Fatal Accidents per Month', font=TITLE_FONT),
    xaxis=dict(tickfont=dict(color=TICK), gridcolor=GRID, title_font=dict(color=TICK)),
    yaxis=dict(title='Fatal Accidents', tickfont=dict(color=TICK), gridcolor=GRID, title_font=dict(color=TICK)),
    height=340, margin=MARGIN
)
charts['monthly'] = fig5.to_json()

# ── 6. Risk factors horizontal bar ───────────────────────────────────────────
risk_items = [
    ('Baseline',                0.821),
    ('Wet road',                0.913),
    ('Motorcycle involved',     0.974),
    ('Speeding suspected',      0.978),
    ('Dark night lighting',     1.106),
    ('Alcohol suspected',       1.119),
    ('No helmet (motorcycle)',  1.250),
]
labels = [r[0] for r in risk_items]
values = [r[1] for r in risk_items]
bar_colors = ['#00d4ff' if v <= 0.821 else ('#ffa502' if v < 1.0 else '#ff4757') for v in values]

fig6 = go.Figure(go.Bar(
    x=values, y=labels, orientation='h',
    marker_color=bar_colors,
    text=[f'{v:.3f}' for v in values],
    textposition='outside',
    textfont=dict(color='#e0e0e0', size=12),
    hovertemplate='<b>%{y}</b><br>Mean severity score: %{x:.3f}<extra></extra>',
))
fig6.add_vline(x=0.821, line_dash='dash', line_color='#00d4ff',
               line_width=1.5, opacity=0.7)
fig6.add_annotation(x=0.821, y=1.04, xref='x', yref='paper',
                    text='Baseline', showarrow=False,
                    font=dict(color='#00d4ff', size=11))
fig6.update_layout(
    template=DARK, paper_bgcolor=BG, plot_bgcolor=BG,
    title=dict(text='Mean Severity Score by Risk Factor  (0 = Damage only … 3 = Fatal)', font=TITLE_FONT),
    xaxis=dict(title='Mean Severity Score', range=[0, 1.65],
               tickfont=dict(color=TICK), gridcolor=GRID, title_font=dict(color=TICK)),
    yaxis=dict(tickfont=dict(color='#e0e0e0', size=12), gridcolor='rgba(0,0,0,0)'),
    height=380, margin=dict(t=55, b=60, l=210, r=90)
)
charts['risk'] = fig6.to_json()

# ── 7. Vehicle fatal rate ─────────────────────────────────────────────────────
veh_data = [
    ('Other',        1.6),
    ('Bus',          4.6),
    ('Car',          5.6),
    ('Minibus/Taxi', 6.3),
    ('Bicycle',      6.7),
    ('Motorcycle',   8.1),
    ('Truck',       10.1),
]
veh_labels = [v[0] for v in veh_data]
veh_vals   = [v[1] for v in veh_data]
veh_colors = ['#00d4ff' if v < 6 else ('#ffa502' if v < 9 else '#ff4757') for v in veh_vals]

fig7 = go.Figure(go.Bar(
    x=veh_vals, y=veh_labels, orientation='h',
    marker_color=veh_colors,
    text=[f'{v}%' for v in veh_vals],
    textposition='outside',
    textfont=dict(color='#e0e0e0', size=12),
    hovertemplate='<b>%{y}</b><br>Fatal rate: %{x:.1f}%<extra></extra>',
))
fig7.update_layout(
    template=DARK, paper_bgcolor=BG, plot_bgcolor=BG,
    title=dict(text='Fatal Rate by Vehicle Type', font=TITLE_FONT),
    xaxis=dict(title='Fatal %', tickfont=dict(color=TICK), gridcolor=GRID, title_font=dict(color=TICK)),
    yaxis=dict(tickfont=dict(color='#e0e0e0', size=12), gridcolor='rgba(0,0,0,0)'),
    height=360, margin=dict(t=55, b=50, l=130, r=90)
)
charts['vehicle'] = fig7.to_json()

# ── 8. Model performance ──────────────────────────────────────────────────────
with open('outputs/models/model_results.json') as f:
    mr = json.load(f)
rf_f1  = [round(mr['Random Forest']['report'][c]['f1-score'], 3) for c in SEV_ORDER]
xgb_f1 = [round(mr['XGBoost']['report'][c]['f1-score'], 3)      for c in SEV_ORDER]

fig8 = go.Figure()
fig8.add_trace(go.Bar(x=SEV_ORDER, y=rf_f1,  name='Random Forest', marker_color='#00d4ff',
                      text=[f'{v:.3f}' for v in rf_f1], textposition='outside',
                      textfont=dict(color='#e0e0e0')))
fig8.add_trace(go.Bar(x=SEV_ORDER, y=xgb_f1, name='XGBoost',       marker_color='#ffa502',
                      text=[f'{v:.3f}' for v in xgb_f1], textposition='outside',
                      textfont=dict(color='#e0e0e0')))
fig8.update_layout(
    template=DARK, barmode='group', paper_bgcolor=BG, plot_bgcolor=BG,
    title=dict(text='Model F1-Score by Severity Class', font=TITLE_FONT),
    xaxis=dict(tickfont=dict(color='#e0e0e0'), gridcolor=GRID),
    yaxis=dict(title='F1-Score', tickfont=dict(color=TICK), gridcolor=GRID,
               title_font=dict(color=TICK), range=[0, 0.75]),
    legend=dict(font=dict(color='#aaa'), bgcolor=BG),
    annotations=[dict(
        x=0.5, y=0.95, xref='paper', yref='paper', showarrow=False,
        text='Random Forest macro F1 = 0.320  |  XGBoost macro F1 = 0.265',
        font=dict(color='#8899aa', size=11),
        bgcolor='rgba(255,255,255,0.04)', bordercolor='#333',
    )],
    height=380, margin=MARGIN
)
charts['model'] = fig8.to_json()

# ── 9. Map (dark basemap) ─────────────────────────────────────────────────────
fatal_df = df[df['severity'] == 'Fatal'].copy()
other_df = df[df['severity'] != 'Fatal'].sample(600, random_state=42).copy()
map_df = pd.concat([fatal_df, other_df])
map_df = map_df[['latitude', 'longitude', 'district', 'vehicle_primary', 'weather', 'severity']].dropna(subset=['latitude', 'longitude'])
map_df = map_df[(map_df['latitude'].between(-3, 0)) & (map_df['longitude'].between(28, 32))]

fig9 = go.Figure()
for sev, color in zip(SEV_ORDER, SEV_COLORS):
    sub = map_df[map_df['severity'] == sev]
    fig9.add_trace(go.Scattermap(
        lat=sub['latitude'], lon=sub['longitude'], name=sev,
        mode='markers',
        marker=dict(
            size=8 if sev == 'Fatal' else 5,
            color=color,
            opacity=0.9 if sev == 'Fatal' else 0.6,
        ),
        customdata=sub[['district', 'vehicle_primary', 'weather']].values,
        hovertemplate=(
            '<b>' + sev + '</b><br>'
            'District: %{customdata[0]}<br>'
            'Vehicle: %{customdata[1]}<br>'
            'Weather: %{customdata[2]}<extra></extra>'
        ),
    ))
fig9.update_layout(
    map=dict(style='carto-darkmatter', center=dict(lat=-1.95, lon=30.06), zoom=8),
    paper_bgcolor=BG,
    title=dict(text='Accident Locations across Rwanda', font=TITLE_FONT),
    legend=dict(font=dict(color='#e0e0e0'), bgcolor='rgba(10,14,26,0.85)',
                bordercolor='#1e2a3a', borderwidth=1),
    height=530, margin=dict(t=55, b=10, l=10, r=10)
)
charts['map'] = fig9.to_json()

with open('/tmp/dashboard_charts.json', 'w') as f:
    json.dump(charts, f)

print('Charts generated:', list(charts.keys()))
print('File written to /tmp/dashboard_charts.json')
