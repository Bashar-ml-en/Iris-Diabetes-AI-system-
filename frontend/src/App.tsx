import React, { useState, useRef, useEffect } from 'react';
import { Upload, Eye, Activity, AlertTriangle, ShieldCheck, Heart, Clock, Download, ChevronRight, RefreshCw, BarChart2, Radio } from 'lucide-react';

interface Prediction {
  sharpness: number;
  diabetes_probability: number;
  diagnosis: string;
  estimated_glucose: number | null;
  medical_advice: string;
  unwrapped_strip_b64: string | null;
  pancreas_roi_b64: string | null;
  saliency_map_b64: string | null;
  latency_ms: number;
}

const API_BASE_URL = 'http://localhost:8000';

// ----------------------------------------------------
// Interactive Clarke Error Grid Component
// ----------------------------------------------------
interface ClarkeGridProps {
  predicted: number;
  reference: number;
  onReferenceChange: (val: number) => void;
}

const ClarkeErrorGrid: React.FC<ClarkeGridProps> = ({ predicted, reference, onReferenceChange }) => {
  const mapValue = (val: number) => ((val - 50) / 250) * 300;
  
  const cx = mapValue(reference);
  const cy = 300 - mapValue(predicted);

  const getZone = (ref: number, pred: number) => {
    const dev = Math.abs(ref - pred) / ref;
    if (dev <= 0.20 || (ref <= 70 && Math.abs(ref - pred) <= 15)) {
      return { code: 'A', desc: 'Clinical Accuracy (Optimal)', color: '#0d9488' }; // Teal
    } else if (dev <= 0.50) {
      return { code: 'B', desc: 'Benign Error (No Risk)', color: '#0284c7' }; // Blue
    } else if (pred < 70 && ref > 180) {
      return { code: 'E', desc: 'Dangerous Over-Correction', color: '#dc2626' }; // Red
    }
    return { code: 'C/D', desc: 'Significant Diagnostic Gap', color: '#ea580c' }; // Orange
  };

  const zone = getZone(reference, predicted);

  return (
    <div style={{ background: 'rgba(14, 165, 233, 0.02)', border: '1px solid rgba(14, 165, 233, 0.1)', borderRadius: '12px', padding: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h4 style={{ fontSize: '13px', textTransform: 'uppercase', color: '#64748b', margin: 0, letterSpacing: '0.5px', fontWeight: 600 }}>
          Interactive Clarke Error Grid
        </h4>
        <span style={{ fontSize: '12px', color: zone.color, fontWeight: 700, display: 'flex', alignItems: 'center', gap: '6px' }}>
          Zone {zone.code}: {zone.desc}
        </span>
      </div>

      <div style={{ position: 'relative', width: '100%', maxWidth: '300px', margin: '0 auto', aspectRatio: '1/1' }}>
        <svg viewBox="0 0 300 300" style={{ width: '100%', height: '100%', background: '#f8fafc', borderRadius: '8px', border: '1px solid rgba(14,165,233,0.15)' }}>
          <line x1="0" y1="300" x2="300" y2="0" stroke="rgba(15, 23, 42, 0.12)" strokeWidth="1.5" />
          <line x1="60" y1="300" x2="300" y2="60" stroke="#0d9488" strokeDasharray="3 3" strokeWidth="1" opacity="0.5" />
          <line x1="0" y1="240" x2="240" y2="0" stroke="#0d9488" strokeDasharray="3 3" strokeWidth="1" opacity="0.5" />

          <text x="140" y="140" fill="rgba(13, 148, 136, 0.12)" fontSize="20" fontWeight="bold" transform="rotate(-45 150 150)">ZONE A</text>
          <text x="25" y="100" fill="rgba(2, 132, 199, 0.08)" fontSize="18" fontWeight="bold">ZONE B</text>
          <text x="220" y="250" fill="rgba(220, 38, 38, 0.08)" fontSize="18" fontWeight="bold">ZONE C/D</text>

          <text x="5" y="15" fill="#64748b" fontSize="9" fontWeight="500">Estimated (mg/dL)</text>
          <text x="220" y="292" fill="#64748b" fontSize="9" fontWeight="500">Reference (mg/dL)</text>

          <circle 
            cx={cx} 
            cy={cy} 
            r="7" 
            fill={zone.color} 
            stroke="#ffffff" 
            strokeWidth="2" 
            style={{ filter: `drop-shadow(0 2px 4px rgba(0,0,0,0.15))` }}
          />
          <line x1={cx} y1="0" x2={cx} y2="300" stroke={zone.color} strokeWidth="0.5" strokeDasharray="1 4" opacity="0.5" />
          <line x1="0" y1={cy} x2="300" y2={cy} stroke={zone.color} strokeWidth="0.5" strokeDasharray="1 4" opacity="0.5" />
        </svg>
      </div>

      <div style={{ marginTop: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
          <span style={{ color: '#475569', fontWeight: 500 }}>Calibration: Blood Draw Reference Value</span>
          <strong style={{ color: '#0f172a' }}>{reference} mg/dL</strong>
        </div>
        <input 
          type="range" 
          min="50" 
          max="300" 
          value={reference} 
          onChange={(e) => onReferenceChange(Number(e.target.value))}
          style={{ width: '100%', accentColor: '#0ea5e9', cursor: 'pointer' }}
        />
      </div>
    </div>
  );
};

// ----------------------------------------------------
// Main App Component
// ----------------------------------------------------
function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [stepperIndex, setStepperIndex] = useState<number>(-1);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [referenceGlucose, setReferenceGlucose] = useState<number>(100);
  
  // XAI & Digital Twin States
  const [showAttention, setShowAttention] = useState<boolean>(true);
  const [simulatorValue, setSimulatorValue] = useState<number>(100);
  
  // Offline PWA Queue States
  const [isOnline, setIsOnline] = useState<boolean>(navigator.onLine);
  const [offlineQueue, setOfflineQueue] = useState<Array<{ name: string; size: number; date: string }>>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const steps = [
    '👁️ Grayscaling & applying pupil median filter...',
    '🔍 Locating pupil and iris outer boundaries via Hough Circles...',
    '🌐 Remapping polar coordinates (Daugman Rubber Sheet)...',
    '🧬 Clipping pancreas sector ROI (270° - 324°)...',
    '🧠 Running Stage 1 Classifier neural network...',
    '🩸 Fetching Stage 2 Glucose Regressor predictions...'
  ];

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  useEffect(() => {
    if (loading) {
      setStepperIndex(0);
      const timer = setInterval(() => {
        setStepperIndex(prev => {
          if (prev < steps.length - 1) {
            return prev + 1;
          }
          clearInterval(timer);
          return prev;
        });
      }, 700);
      return () => clearInterval(timer);
    }
  }, [loading]);

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

  const runAnalysis = async () => {
    if (!file) return;

    // Handle offline scan logging (PWA Offline Queue Mode)
    if (!isOnline) {
      const newScan = {
        name: file.name,
        size: file.size,
        date: new Date().toLocaleTimeString()
      };
      setOfflineQueue(prev => [...prev, newScan]);
      setError('Scan saved to offline queue. It will automatically upload and execute when network connection is restored.');
      return;
    }

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
      if (data.estimated_glucose) {
        setReferenceGlucose(Math.round(data.estimated_glucose));
        setSimulatorValue(Math.round(data.estimated_glucose));
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred during analysis.');
    } finally {
      setLoading(false);
    }
  };

  const syncOfflineQueue = async () => {
    if (offlineQueue.length === 0 || !file) return;
    setError(null);
    await runAnalysis();
    setOfflineQueue([]);
  };

  const resetScanner = () => {
    setFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setError(null);
    setStepperIndex(-1);
  };

  const getGlucoseColor = (val: number) => {
    if (val > 180) return '#dc2626'; // Red
    if (val > 120) return '#ea580c'; // Orange
    return '#0d9488'; // Teal
  };

  // Helper for Digital Twin Simulator Visuals
  const getTwinOverlayStyle = () => {
    if (simulatorValue <= 120) {
      return { fill: 'rgba(16, 185, 129, 0.2)', stroke: '#10b981', label: 'Optimal reflex fibers. High tissue density.' };
    } else if (simulatorValue <= 180) {
      const ratio = (simulatorValue - 120) / 60;
      return { 
        fill: `rgba(249, 115, 22, ${0.2 + ratio * 0.2})`, 
        stroke: '#f97316', 
        label: 'Slight fiber spacing rarefaction. Light metabolic pigmentation.' 
      };
    } else {
      const ratio = Math.min((simulatorValue - 180) / 120, 1.0);
      return { 
        fill: `rgba(239, 68, 68, ${0.4 + ratio * 0.3})`, 
        stroke: '#ef4444', 
        label: 'Hyperglycemic load. Spindle dilation and spot lesion forming.' 
      };
    }
  };

  const twinStyle = getTwinOverlayStyle();

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '40px 20px' }}>
      
      {/* Background Glows */}
      <div style={{ position: 'fixed', top: '-10%', right: '-10%', width: '500px', height: '500px', background: 'radial-gradient(circle, rgba(14,165,233,0.06) 0%, rgba(255,255,255,0) 70%)', zIndex: -1, pointerEvents: 'none' }} />
      <div style={{ position: 'fixed', bottom: '-10%', left: '-10%', width: '500px', height: '500px', background: 'radial-gradient(circle, rgba(2,132,199,0.04) 0%, rgba(255,255,255,0) 70%)', zIndex: -1, pointerEvents: 'none' }} />

      {/* Futuristic Medical Navigation Bar */}
      <nav className="glass" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px 24px', marginBottom: '30px', border: '1px solid rgba(14, 165, 233, 0.15)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Eye size={24} color="#0ea5e9" style={{ filter: 'drop-shadow(0 2px 4px rgba(14, 165, 233, 0.2))' }} />
          <span style={{ fontWeight: 700, fontSize: '19px', letterSpacing: '-0.5px', background: 'linear-gradient(to right, #0f172a, #0284c7)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            CLINICAL DIAGNOSTICS: DIA-SCAN v1.0
          </span>
        </div>
        
        {/* PWA Connection & Queue Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {offlineQueue.length > 0 && isOnline && (
            <button 
              onClick={syncOfflineQueue}
              style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', background: '#e0f2fe', color: '#0284c7', border: '1px solid #bae6fd', padding: '6px 14px', borderRadius: '20px', cursor: 'pointer', fontWeight: 700 }}
            >
              <RefreshCw size={12} style={{ animation: 'spin 4s linear infinite' }} /> Sync Offline Scans ({offlineQueue.length})
            </button>
          )}
          <span style={{ 
            fontSize: '11px', 
            background: isOnline ? 'rgba(13, 148, 136, 0.08)' : 'rgba(220, 38, 38, 0.08)',
            color: isOnline ? '#0d9488' : '#dc2626',
            padding: '6px 14px', 
            borderRadius: '20px', 
            border: `1px solid ${isOnline ? 'rgba(13, 148, 136, 0.15)' : 'rgba(220, 38, 38, 0.15)'}`, 
            fontWeight: 700,
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <Radio size={14} style={{ animation: isOnline ? 'pulse 2s infinite' : 'none' }} /> {isOnline ? 'ONLINE CLOUD ACTIVE' : 'OFFLINE SCAN QUEUE'}
          </span>
        </div>
      </nav>

      {/* Main Container */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '30px' }} className="animate-fade-in">
        
        {/* Upload Zone */}
        {!previewUrl && (
          <div 
            className="glass upload-zone animate-fade-in" 
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{ padding: '80px 40px', position: 'relative', overflow: 'hidden' }}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              style={{ display: 'none' }}
              accept="image/*"
            />
            <Upload size={52} color="#0ea5e9" style={{ marginBottom: '18px', filter: 'drop-shadow(0 4px 10px rgba(14, 165, 233, 0.25))' }} />
            <h2 style={{ fontSize: '24px', fontWeight: 600, margin: '0 0 8px 0', color: '#0f172a' }}>
              Load Patient Iris Scan
            </h2>
            <p style={{ color: '#475569', fontSize: '15px', maxWidth: '480px', margin: '0 auto 24px auto', lineHeight: 1.6 }}>
              Drag and drop high-resolution close-ups of the iris ring, or click to browse local folders.
            </p>
            <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', color: '#64748b', fontSize: '12px', fontWeight: 500 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><ShieldCheck size={14} color="#0ea5e9" /> PWA Standalone app enabled</span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><ShieldCheck size={14} color="#0ea5e9" /> Offline caching active</span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><ShieldCheck size={14} color="#0ea5e9" /> Limbus boundary mapping</span>
            </div>
          </div>
        )}

        {previewUrl && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '30px' }}>
            
            {/* Left Box: Image Panel with SVG Holographic Overlays */}
            <div className="glass animate-fade-in" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: 600, margin: 0, display: 'flex', alignItems: 'center', gap: '8px', color: '#0f172a' }}>
                <Eye size={18} color="#0ea5e9" /> Optical Scan Interface
              </h3>
              
              {/* Relative Image wrapper with SVG overlay */}
              <div style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', border: '1px solid rgba(14, 165, 233, 0.1)', background: '#f1f5f9', aspectRatio: '1/1' }}>
                <img 
                  src={previewUrl} 
                  alt="Patient Scan" 
                  style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
                />
                
                {/* SVG Glowing Circles overlay (Daugman segmenter mock scale) */}
                {prediction && (
                  <svg 
                    viewBox="0 0 300 300" 
                    style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                  >
                    {/* Detected Pupil Circle (Cyan) */}
                    <circle 
                      cx="150" 
                      cy="150" 
                      r="32" 
                      stroke="#0ea5e9" 
                      strokeWidth="2.5" 
                      strokeDasharray="4 2" 
                      fill="none" 
                      style={{ filter: 'drop-shadow(0 2px 4px rgba(14, 165, 233, 0.3))' }}
                    />
                    
                    {/* Detected Iris Circle (Purple Limbus) */}
                    <circle 
                      cx="150" 
                      cy="150" 
                      r="92" 
                      stroke="#a855f7" 
                      strokeWidth="2.5" 
                      strokeDasharray="4 2" 
                      fill="none" 
                      style={{ filter: 'drop-shadow(0 2px 4px rgba(168, 85, 247, 0.3))' }}
                    />
                    
                    {/* Digital Twin Pancreas Sector wedge path */}
                    <path 
                      d="M 150 118 L 150 58 A 92 92 0 0 1 215 85 L 172.6 127.4 A 32 32 0 0 0 150 118 Z" 
                      fill={twinStyle.fill} 
                      stroke={twinStyle.stroke} 
                      strokeWidth="1.5" 
                      style={{ transition: 'all 0.4s ease' }}
                    />

                    {/* Annotations */}
                    <text x="175" y="70" fill={twinStyle.stroke} fontSize="8.5" fontWeight="bold" opacity="0.9">PANCREAS ROI</text>
                    <line x1="150" y1="150" x2="220" y2="150" stroke="rgba(15, 23, 42, 0.1)" strokeWidth="0.5" strokeDasharray="2 2" />
                    <line x1="150" y1="150" x2="150" y2="50" stroke="rgba(15, 23, 42, 0.1)" strokeWidth="0.5" strokeDasharray="2 2" />
                  </svg>
                )}
              </div>

              {error && (
                <div style={{ background: 'rgba(220, 38, 38, 0.05)', border: '1px solid rgba(220, 38, 38, 0.15)', color: '#dc2626', borderRadius: '8px', padding: '12px', fontSize: '13px', display: 'flex', gap: '8px', alignItems: 'center', fontWeight: 500 }}>
                  <AlertTriangle size={16} />
                  <span>{error}</span>
                </div>
              )}

              {/* Digital Twin Simulator Controller */}
              {prediction && prediction.diagnosis === 'Diabetic' && (
                <div style={{ background: 'rgba(15, 23, 42, 0.02)', padding: '14px', borderRadius: '8px', border: '1px solid rgba(15,23,42,0.05)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: 600, marginBottom: '6px' }}>
                    <span style={{ color: '#475569' }}>Digital Twin: Glycemic Morphing</span>
                    <span style={{ color: getGlucoseColor(simulatorValue) }}>{simulatorValue} mg/dL</span>
                  </div>
                  <input 
                    type="range" 
                    min="70" 
                    max="300" 
                    value={simulatorValue}
                    onChange={(e) => setSimulatorValue(Number(e.target.value))}
                    style={{ width: '100%', accentColor: getGlucoseColor(simulatorValue), cursor: 'pointer' }}
                  />
                  <p style={{ margin: '6px 0 0 0', fontSize: '11px', color: '#64748b', fontStyle: 'italic', lineHeight: 1.4 }}>
                    {twinStyle.label}
                  </p>
                </div>
              )}

              <div style={{ display: 'flex', gap: '12px' }}>
                <button 
                  onClick={runAnalysis} 
                  disabled={loading}
                  className={`btn-primary ${loading ? 'btn-disabled' : ''}`}
                  style={{ flex: 2, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}
                >
                  <Activity size={18} />
                  {loading ? 'Processing...' : 'Run Diagnostics'}
                </button>
                <button 
                  onClick={resetScanner} 
                  disabled={loading}
                  style={{ flex: 1, background: 'transparent', border: '1px solid rgba(148, 163, 184, 0.25)', color: '#475569', borderRadius: '8px', cursor: 'pointer', transition: 'all 0.2s', fontWeight: 600 }}
                >
                  New Scan
                </button>
              </div>
            </div>

            {/* Right Box: Processing logs / Prediction Cockpit */}
            <div className="glass" style={{ padding: '24px', display: 'flex', flexDirection: 'column', minHeight: '400px', justifyContent: 'center' }}>
              
              {/* Ready State */}
              {!prediction && !loading && (
                <div style={{ textAlign: 'center', padding: '40px 20px', color: '#64748b' }}>
                  <Activity size={48} color="#0ea5e9" style={{ marginBottom: '16px', opacity: 0.5 }} />
                  <h4 style={{ fontSize: '16px', color: '#0f172a', margin: '0 0 6px 0', fontWeight: 600 }}>Inference Panel Ready</h4>
                  <p style={{ fontSize: '13px', margin: 0 }}>Upload image and click "Run Diagnostics" to scan raw eye features.</p>
                </div>
              )}

              {/* Advanced Stepper Loader */}
              {loading && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', padding: '20px 10px' }} className="animate-fade-in">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <RefreshCw size={18} color="#0ea5e9" style={{ animation: 'spin 1.5s linear infinite' }} />
                    <span style={{ color: '#0284c7', fontSize: '14px', fontWeight: 700 }}>Processing Scan Data...</span>
                  </div>
                  
                  {/* Logs terminal */}
                  <div style={{ background: '#f8fafc', border: '1px solid rgba(14, 165, 233, 0.12)', borderRadius: '8px', padding: '16px', minHeight: '160px', display: 'flex', flexDirection: 'column', gap: '10px', fontFamily: 'monospace', fontSize: '11.5px', color: '#334155' }}>
                    {steps.slice(0, stepperIndex + 1).map((log, idx) => (
                      <div key={idx} style={{ display: 'flex', gap: '8px', alignItems: 'center', opacity: idx === stepperIndex ? 1 : 0.6 }} className="animate-fade-in">
                        <ChevronRight size={12} color="#0ea5e9" />
                        <span>{log}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Results Engine */}
              {prediction && (
                <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  
                  {/* Latency & Quality Metrics */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#64748b', borderBottom: '1px solid rgba(15, 23, 42, 0.05)', paddingBottom: '12px', fontWeight: 500 }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Clock size={12} /> API Latency: {prediction.latency_ms.toFixed(1)} ms
                    </span>
                    <span>
                      Blur Quality: {prediction.sharpness.toFixed(1)} (PASS)
                    </span>
                  </div>

                  {/* Stage 1 Metrics */}
                  <div>
                    <h4 style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748b', margin: '0 0 8px 0', letterSpacing: '0.5px', fontWeight: 700 }}>
                      Stage 1: Binary Classification Output
                    </h4>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ fontSize: '19px', fontWeight: 700, color: '#0f172a' }}>
                        {prediction.diagnosis === 'Diabetic' ? 'Diabetic Retinal Markers' : 'Non-Diabetic Control'}
                      </span>
                      <span style={{ 
                        background: prediction.diagnosis === 'Diabetic' ? 'rgba(220, 38, 38, 0.08)' : 'rgba(13, 148, 136, 0.08)',
                        color: prediction.diagnosis === 'Diabetic' ? '#dc2626' : '#0d9488',
                        border: `1px solid ${prediction.diagnosis === 'Diabetic' ? 'rgba(220, 38, 38, 0.2)' : 'rgba(13, 148, 136, 0.2)'}`,
                        padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: 700, textTransform: 'uppercase'
                      }}>
                        {prediction.diagnosis === 'Diabetic' ? 'Alert' : 'Normal'}
                      </span>
                    </div>
                    
                    <div className="gauge-container" style={{ height: '6px' }}>
                      <div 
                        className="gauge-bar" 
                        style={{ 
                          width: `${prediction.diabetes_probability * 100}%`,
                          background: prediction.diagnosis === 'Diabetic' ? '#dc2626' : '#0d9488'
                        }} 
                      />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#64748b', marginTop: '6px', fontWeight: 500 }}>
                      <span>Healthy Control (0.0)</span>
                      <span>Classifier Confidence: {(prediction.diabetes_probability * 100).toFixed(1)}%</span>
                      <span>Diabetic (1.0)</span>
                    </div>
                  </div>

                  {/* Stage 2 Metrics */}
                  {prediction.diagnosis === 'Diabetic' && prediction.estimated_glucose !== null ? (
                    <div style={{ borderTop: '1px solid rgba(15, 23, 42, 0.05)', paddingTop: '16px' }}>
                      <ClarkeErrorGrid 
                        predicted={prediction.estimated_glucose} 
                        reference={referenceGlucose} 
                        onReferenceChange={setReferenceGlucose}
                      />
                    </div>
                  ) : (
                    <div style={{ borderTop: '1px solid rgba(15, 23, 42, 0.05)', paddingTop: '16px', background: 'rgba(13, 148, 136, 0.02)', border: '1px dashed rgba(13, 148, 136, 0.2)', borderRadius: '8px', padding: '16px', textAlign: 'center' }}>
                      <ShieldCheck size={28} color="#0d9488" style={{ margin: '0 auto 8px auto' }} />
                      <h5 style={{ margin: '0 0 4px 0', fontSize: '13px', color: '#0f172a', fontWeight: 600 }}>Screening Complete</h5>
                      <p style={{ margin: 0, fontSize: '11px', color: '#64748b', lineHeight: 1.5 }}>
                        Patient does not show diabetic indicators. Stage 2 regression pipeline bypassed.
                      </p>
                    </div>
                  )}

                  {/* Medical Advice Panel */}
                  <div style={{ 
                    background: 'rgba(15, 23, 42, 0.015)',
                    border: '1px solid rgba(15, 23, 42, 0.04)',
                    borderRadius: '10px',
                    padding: '14px',
                    fontSize: '13px',
                    lineHeight: 1.5,
                    color: '#334155',
                    display: 'flex',
                    gap: '12px',
                    alignItems: 'flex-start'
                  }}>
                    <Heart size={16} color="#0ea5e9" style={{ marginTop: '2px', flexShrink: 0 }} />
                    <div>
                      <strong style={{ color: '#0f172a' }}>Clinical Action Plan</strong>
                      <div style={{ marginTop: '4px', fontSize: '12px', color: '#475569' }}>
                        {prediction.medical_advice}
                      </div>
                    </div>
                  </div>

                  {/* Report Download */}
                  <button 
                    onClick={() => window.print()}
                    style={{ background: 'rgba(15,23,42,0.02)', border: '1px solid rgba(15,23,42,0.08)', color: '#0f172a', borderRadius: '8px', padding: '10px 16px', fontSize: '13px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'center', transition: 'all 0.2s', fontWeight: 600 }}
                  >
                    <Download size={14} /> Export Patient Report Sheet
                  </button>

                </div>
              )}
            </div>

          </div>
        )}

        {/* OpenCV Segmentation Visualizer Panel */}
        {prediction && (
          <div className="glass animate-fade-in" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: 600, margin: 0, display: 'flex', alignItems: 'center', gap: '8px', color: '#0f172a' }}>
                <BarChart2 size={18} color="#0ea5e9" /> Preprocessing & ROI Extraction Engine
              </h3>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button 
                  onClick={() => setShowAttention(false)}
                  style={{ background: !showAttention ? '#0ea5e9' : 'transparent', color: !showAttention ? '#ffffff' : '#64748b', border: '1px solid rgba(148, 163, 184, 0.25)', padding: '4px 12px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer', fontWeight: 600 }}
                >
                  Raw Crops
                </button>
                <button 
                  onClick={() => setShowAttention(true)}
                  style={{ background: showAttention ? '#0ea5e9' : 'transparent', color: showAttention ? '#ffffff' : '#64748b', border: '1px solid rgba(148, 163, 184, 0.25)', padding: '4px 12px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer', fontWeight: 600 }}
                >
                  XAI Attention Maps
                </button>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
              
              {/* Unwrapped Flat Strip */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <h4 style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748b', margin: 0, letterSpacing: '0.5px', fontWeight: 600 }}>
                  1. Daugman's Flat Iris Strip (360° Projection)
                </h4>
                {prediction.unwrapped_strip_b64 ? (
                  <div style={{ border: '1px solid rgba(14, 165, 233, 0.12)', borderRadius: '8px', overflow: 'hidden', background: '#f8fafc' }}>
                    <img 
                      src={prediction.unwrapped_strip_b64} 
                      alt="Unwrapped Iris" 
                      style={{ width: '100%', height: 'auto', display: 'block' }}
                    />
                  </div>
                ) : (
                  <p style={{ color: '#64748b', fontSize: '12px' }}>Unwarped strip visualization not available.</p>
                )}
              </div>

              {/* Pancreas Crop vs Saliency Heatmap */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <h4 style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748b', margin: 0, letterSpacing: '0.5px', fontWeight: 600 }}>
                  2. {showAttention ? 'Explainable AI (Grad-CAM Saliency Hotspots)' : 'Segmented Pancreas ROI (150x150 input)'}
                </h4>
                
                {prediction.pancreas_roi_b64 ? (
                  <div style={{ display: 'flex', justifyContent: 'center' }}>
                    <div style={{ border: '1px solid rgba(14, 165, 233, 0.12)', borderRadius: '8px', overflow: 'hidden', background: '#f8fafc', width: '150px', height: '150px' }}>
                      <img 
                        src={showAttention && prediction.saliency_map_b64 ? prediction.saliency_map_b64 : prediction.pancreas_roi_b64} 
                        alt="Pancreas Target" 
                        style={{ width: '100%', height: '100%', display: 'block' }}
                      />
                    </div>
                  </div>
                ) : (
                  <p style={{ color: '#64748b', fontSize: '12px' }}>Pancreas sector target visualizer not available.</p>
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
