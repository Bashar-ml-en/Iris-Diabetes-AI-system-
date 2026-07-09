import React, { useState, useRef } from 'react';
import { Upload, Eye, Activity, AlertTriangle, ShieldCheck, Heart, Clock } from 'lucide-react';

interface Prediction {
  sharpness: number;
  diabetes_probability: number;
  diagnosis: string;
  estimated_glucose: number | null;
  medical_advice: string;
  unwrapped_strip_b64: string | null;
  pancreas_roi_b64: string | null;
  latency_ms: number;
}

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
      setPrediction(null);
      setError(null);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selected = e.dataTransfer.files[0];
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
      setPrediction(null);
      setError(null);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const runAnalysis = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errDetail = await response.json();
        throw new Error(errDetail.detail || 'Analysis failed. Please check image quality.');
      }

      const data: Prediction = await response.json();
      setPrediction(data);
    } catch (err: any) {
      setError(err.message || 'An error occurred during analysis.');
    } finally {
      setLoading(false);
    }
  };

  const resetScanner = () => {
    setFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setError(null);
  };

  // Helper for color-coding glucose levels
  const getGlucoseColor = (val: number) => {
    if (val > 180) return '#ef4444'; // Red
    if (val > 120) return '#f97316'; // Orange
    return '#10b981'; // Green
  };

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', padding: '40px 20px' }}>
      
      {/* Header */}
      <header style={{ textAlign: 'center', marginBottom: '40px' }} className="animate-fade-in">
        <div style={{ display: 'inline-flex', padding: '12px', background: 'rgba(168, 85, 247, 0.1)', borderRadius: '50%', marginBottom: '16px' }}>
          <Eye size={40} color="#a855f7" />
        </div>
        <h1 style={{ fontSize: '38px', fontWeight: 700, margin: '0 0 10px 0', letterSpacing: '-0.5px' }}>
          AI Iridology Clinical Screening
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '16px', margin: 0 }}>
          High-resolution cascaded medical diagnostics pipeline for diabetes detection and glucose estimation.
        </p>
      </header>

      {/* Main Container */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '30px' }} className="animate-fade-in">
        
        {/* Upload and Input Section */}
        {!previewUrl ? (
          <div 
            className="glass upload-zone" 
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={triggerFileInput}
            style={{ padding: '60px 40px' }}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              style={{ display: 'none' }}
              accept="image/*"
            />
            <Upload size={48} color="#a855f7" style={{ marginBottom: '16px', opacity: 0.8 }} />
            <h3 style={{ fontSize: '20px', fontWeight: 500, margin: '0 0 8px 0' }}>Upload Patient Iris Image</h3>
            <p style={{ color: '#64748b', fontSize: '14px', margin: '0 0 20px 0' }}>
              Drag and drop high-resolution iris photography here, or click to browse files
            </p>
            <div style={{ color: '#94a3b8', fontSize: '12px', display: 'flex', gap: '12px', justifyContent: 'center' }}>
              <span>• JPEG or PNG</span>
              <span>• Min resolution 300x300</span>
              <span>• Low-blur closeups</span>
            </div>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '30px' }}>
            
            {/* Left: Input Image Preview & Control */}
            <div className="glass" style={{ padding: '24px' }}>
              <h3 style={{ fontSize: '18px', fontWeight: 600, margin: '0 0 16px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Eye size={20} color="#a855f7" /> Patient Scan Input
              </h3>
              
              <div style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.05)', background: '#12141c', marginBottom: '20px' }}>
                <img 
                  src={previewUrl} 
                  alt="Patient Iris" 
                  style={{ width: '100%', height: 'auto', display: 'block', maxHeight: '400px', objectFit: 'contain' }}
                />
              </div>

              {error && (
                <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', color: '#f87171', borderRadius: '8px', padding: '12px', fontSize: '14px', marginBottom: '20px', display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <AlertTriangle size={18} />
                  <span>{error}</span>
                </div>
              )}

              <div style={{ display: 'flex', gap: '12px' }}>
                <button 
                  onClick={runAnalysis} 
                  disabled={loading}
                  className={`btn-primary ${loading ? 'btn-disabled' : ''}`}
                  style={{ flex: 2 }}
                >
                  {loading ? 'Analyzing...' : 'Run Clinical Analysis'}
                </button>
                <button 
                  onClick={resetScanner} 
                  disabled={loading}
                  style={{ flex: 1, background: 'transparent', border: '1px solid rgba(255, 255, 255, 0.1)', color: '#94a3b8', borderRadius: '8px', cursor: 'pointer' }}
                >
                  Cancel
                </button>
              </div>
            </div>

            {/* Right: Results Display */}
            <div className="glass" style={{ padding: '24px', display: 'flex', flexDirection: 'column', minHeight: '400px', justifyContent: 'center' }}>
              {!prediction && !loading && (
                <div style={{ textAlign: 'center', color: '#64748b' }}>
                  <Activity size={40} style={{ marginBottom: '12px', opacity: 0.5 }} />
                  <p>Ready to analyze. Click "Run Clinical Analysis" to start the two-stage pipeline.</p>
                </div>
              )}

              {loading && (
                <div style={{ textAlign: 'center' }}>
                  <div style={{ border: '3px solid rgba(168, 85, 247, 0.1)', borderTop: '3px solid #a855f7', borderRadius: '50%', width: '40px', height: '40px', animation: 'spin 1s linear infinite', margin: '0 auto 16px auto' }} />
                  <style>{`
                    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                  `}</style>
                  <p style={{ color: '#94a3b8' }}>Segmenting Iris & unwrapping circular structures...</p>
                </div>
              )}

              {prediction && (
                <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  
                  {/* Latency & Quality Metrics */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#64748b', borderBottom: '1px solid rgba(255, 255, 255, 0.05)', paddingBottom: '12px' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Clock size={14} /> Latency: {prediction.latency_ms.toFixed(1)} ms
                    </span>
                    <span>
                      Sharpness Score: {prediction.sharpness.toFixed(1)}
                    </span>
                  </div>

                  {/* Stage 1 Results */}
                  <div>
                    <h4 style={{ fontSize: '13px', textTransform: 'uppercase', color: '#94a3b8', margin: '0 0 10px 0', letterSpacing: '0.5px' }}>
                      Stage 1: Binary Classification
                    </h4>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ fontSize: '18px', fontWeight: 600 }}>
                        {prediction.diagnosis === 'Diabetic' ? 'Diabetic Markers Detected' : 'No Diabetes Markers'}
                      </span>
                      <span style={{ 
                        background: prediction.diagnosis === 'Diabetic' ? 'rgba(249, 115, 22, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                        color: prediction.diagnosis === 'Diabetic' ? '#f97316' : '#10b981',
                        padding: '4px 10px', borderRadius: '6px', fontSize: '13px', fontWeight: 500
                      }}>
                        {prediction.diagnosis === 'Diabetic' ? 'Alert' : 'Normal'}
                      </span>
                    </div>
                    <div className="gauge-container">
                      <div 
                        className="gauge-bar" 
                        style={{ 
                          width: `${prediction.diabetes_probability * 100}%`,
                          background: prediction.diagnosis === 'Diabetic' ? '#f97316' : '#10b981'
                        }} 
                      />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#64748b', marginTop: '6px' }}>
                      <span>Control (0.0)</span>
                      <span>Diabetic Confidence: {(prediction.diabetes_probability * 100).toFixed(1)}%</span>
                      <span>Diabetic (1.0)</span>
                    </div>
                  </div>

                  {/* Stage 2 Results */}
                  {prediction.diagnosis === 'Diabetic' && prediction.estimated_glucose !== null && (
                    <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.05)', paddingTop: '20px' }}>
                      <h4 style={{ fontSize: '13px', textTransform: 'uppercase', color: '#94a3b8', margin: '0 0 10px 0', letterSpacing: '0.5px' }}>
                        Stage 2: Continuous Regression (EasyGlucose)
                      </h4>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '8px' }}>
                        <span style={{ fontSize: '14px', color: '#cbd5e1' }}>Estimated Blood Glucose</span>
                        <span style={{ fontSize: '26px', fontWeight: 700, color: getGlucoseColor(prediction.estimated_glucose) }}>
                          {prediction.estimated_glucose.toFixed(1)} <span style={{ fontSize: '14px', fontWeight: 400 }}>mg/dL</span>
                        </span>
                      </div>
                      
                      {/* Glucose Gauge */}
                      <div className="gauge-container" style={{ margin: '14px 0' }}>
                        <div 
                          className="gauge-marker" 
                          style={{ 
                            left: `${Math.min(Math.max(((prediction.estimated_glucose - 50) / 250) * 100, 0), 100)}%`,
                            borderColor: getGlucoseColor(prediction.estimated_glucose)
                          }} 
                        />
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#64748b' }}>
                        <span>Hypoglycemia (&lt;70)</span>
                        <span>Normal (70-120)</span>
                        <span>Hyperglycemia (&gt;180)</span>
                      </div>
                    </div>
                  )}

                  {/* Medical Advice Box */}
                  <div style={{ 
                    background: 'rgba(255, 255, 255, 0.02)',
                    border: '1px solid rgba(255, 255, 255, 0.05)',
                    borderRadius: '10px',
                    padding: '16px',
                    fontSize: '14px',
                    lineHeight: 1.6,
                    color: '#cbd5e1',
                    display: 'flex',
                    gap: '12px',
                    alignItems: 'flex-start'
                  }}>
                    <div style={{ marginTop: '2px' }}>
                      {prediction.diagnosis === 'Control' ? (
                        <ShieldCheck size={20} color="#10b981" />
                      ) : prediction.estimated_glucose && prediction.estimated_glucose > 180 ? (
                        <AlertTriangle size={20} color="#ef4444" />
                      ) : (
                        <Heart size={20} color="#f97316" />
                      )}
                    </div>
                    <div>
                      <strong>Advice Summary:</strong>
                      <div style={{ marginTop: '4px', fontSize: '13px', color: '#94a3b8' }}>
                        {prediction.medical_advice}
                      </div>
                    </div>
                  </div>

                </div>
              )}
            </div>

          </div>
        )}

        {/* OpenCV Visualization section */}
        {prediction && (
          <div className="glass animate-fade-in" style={{ padding: '24px', marginTop: '10px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: 600, margin: '0 0 20px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={20} color="#a855f7" /> Automated Segmentation Visualizer
            </h3>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
              
              {/* Unwrapped Iris */}
              <div>
                <h4 style={{ fontSize: '13px', textTransform: 'uppercase', color: '#94a3b8', margin: '0 0 10px 0', letterSpacing: '0.5px' }}>
                  1. Daugman's Flat Iris Strip (360° Projection)
                </h4>
                {prediction.unwrapped_strip_b64 ? (
                  <div style={{ border: '1px solid rgba(255, 255, 255, 0.05)', borderRadius: '8px', overflow: 'hidden', background: '#12141c' }}>
                    <img 
                      src={prediction.unwrapped_strip_b64} 
                      alt="Unwrapped Iris" 
                      style={{ width: '100%', height: 'auto', display: 'block' }}
                    />
                  </div>
                ) : (
                  <p style={{ color: '#64748b', fontSize: '13px' }}>Unwarping visual not available.</p>
                )}
              </div>

              {/* Pancreas Crop */}
              <div>
                <h4 style={{ fontSize: '13px', textTransform: 'uppercase', color: '#94a3b8', margin: '0 0 10px 0', letterSpacing: '0.5px' }}>
                  2. Segmented Pancreas Region of Interest (150x150 ROI)
                </h4>
                {prediction.pancreas_roi_b64 ? (
                  <div style={{ display: 'flex', justifyContent: 'center' }}>
                    <div style={{ border: '1px solid rgba(255, 255, 255, 0.05)', borderRadius: '8px', overflow: 'hidden', background: '#12141c', width: '150px', height: '150px' }}>
                      <img 
                        src={prediction.pancreas_roi_b64} 
                        alt="Pancreas ROI" 
                        style={{ width: '100%', height: '100%', display: 'block' }}
                      />
                    </div>
                  </div>
                ) : (
                  <p style={{ color: '#64748b', fontSize: '13px' }}>Pancreas ROI visual not available.</p>
                )}
              </div>

            </div>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
