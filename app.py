import gradio as gr

def triage(heart_rate, oxygen, blood_pressure):
    # Calculate risk
    if heart_rate > 120 or oxygen < 90 or blood_pressure < 90:
        result = "🚨 HIGH RISK - Immediate Escalation Needed"
        color = "red"
    elif heart_rate > 100 or oxygen < 94 or blood_pressure < 100:
        result = "⚠️ MEDIUM RISK - Treatment Required"
        color = "orange"
    else:
        result = "✅ LOW RISK - Can be Discharged"
        color = "green"
    
    return f"""
    <div style="padding: 20px; border-radius: 10px; background: {color}20; border: 2px solid {color};">
        <h2 style="color: {color};">Triage Assessment</h2>
        <p><strong>Heart Rate:</strong> {heart_rate} bpm</p>
        <p><strong>Oxygen:</strong> {oxygen}%</p>
        <p><strong>Blood Pressure:</strong> {blood_pressure} mmHg</p>
        <h3>{result}</h3>
    </div>
    """

demo = gr.Interface(
    fn=triage,
    inputs=[
        gr.Slider(40, 200, label="Heart Rate (bpm)", value=80),
        gr.Slider(70, 100, label="Oxygen Saturation (%)", value=96),
        gr.Slider(50, 200, label="Blood Pressure (systolic)", value=120)
    ],
    outputs=gr.HTML(),
    title="🏥 Medical Triage AI Agent",
    description="Emergency Department Triage System"
)

demo.launch(server_name="0.0.0.0", server_port=7860)