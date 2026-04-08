import gradio as gr

def triage_patient(age, heart_rate, oxygen_saturation, systolic_bp, temperature):
    # Simple NEWS score calculation
    news_score = 0
    
    # Heart rate
    if heart_rate <= 40 or heart_rate >= 131:
        news_score += 3
    elif heart_rate <= 50 or heart_rate >= 111:
        news_score += 2
    elif heart_rate <= 55 or heart_rate >= 106:
        news_score += 1
    
    # Oxygen
    if oxygen_saturation <= 91:
        news_score += 3
    elif oxygen_saturation <= 93:
        news_score += 2
    elif oxygen_saturation <= 95:
        news_score += 1
    
    # Temperature
    if temperature <= 35.0 or temperature >= 39.1:
        news_score += 2
    elif temperature >= 38.1:
        news_score += 1
    
    # Blood pressure
    if systolic_bp <= 90:
        news_score += 3
    elif systolic_bp <= 100:
        news_score += 2
    
    # Determine action
    if news_score <= 1:
        risk = "LOW"
        action = "✅ DISCHARGE"
        color = "green"
    elif news_score <= 5:
        risk = "MEDIUM"
        action = "💊 TREAT"
        color = "orange"
    else:
        risk = "HIGH"
        action = "🚨 ESCALATE to ICU"
        color = "red"
    
    # Return formatted result
    return f"""
    <div style="font-family: Arial; padding: 20px; border: 2px solid {color}; border-radius: 10px;">
        <h2>🏥 Triage Results</h2>
        <p><strong>NEWS Score:</strong> {news_score}</p>
        <p><strong>Risk Level:</strong> <span style="color: {color};">{risk}</span></p>
        <p><strong>Recommended Action:</strong> {action}</p>
        <hr>
        <p><em>Patient Age: {age} | HR: {heart_rate} | O2: {oxygen_saturation}% | BP: {systolic_bp} | Temp: {temperature}°C</em></p>
    </div>
    """

# Create the Gradio interface
demo = gr.Interface(
    fn=triage_patient,
    inputs=[
        gr.Slider(0, 120, label="Age", value=65),
        gr.Slider(0, 200, label="Heart Rate", value=80),
        gr.Slider(50, 100, label="Oxygen Saturation (%)", value=96),
        gr.Slider(50, 200, label="Systolic BP", value=120),
        gr.Slider(35, 41, label="Temperature (°C)", value=37.0)
    ],
    outputs=gr.HTML(label="Assessment"),
    title="🏥 Medical Triage AI Agent",
    description="Emergency Department Triage System using NEWS Score"
)

# This is REQUIRED for Hugging Face Spaces
demo.launch(server_name="0.0.0.0", server_port=7860)