import React, { useState, useRef, useEffect } from 'react';
import { Upload, Eye, Activity, AlertTriangle, ShieldCheck, Heart, Clock, Download, ChevronRight, RefreshCw, BarChart2, Radio, CheckCircle, Server, Cpu, HelpCircle, AlertCircle } from 'lucide-react';

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
  const [activeNodeIndex, setActiveNodeIndex] = useState<number>(-1);
  const [selectedNodeDetails, setSelectedNodeDetails] = useState<string | null>(null);
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

  // System Architecture Pipeline Nodes
  const architectureNodes = [
    { title: "Image Upload", desc: "Optic Scan Input", icon: Upload, details: "Accepts high-resolution close-up PNG/JPEG ocular captures." },
    { title: "Quality Guard", desc: "Laplacian check", icon: ShieldCheck, details: "Calculates mathematical edge sharpness. Filters out blurry images." },
    { title: "Daugman Remap", desc: "Segment & Unwarp", icon: Eye, details: "Hough Circles pupil tracking and polar rubber-sheet unwarping." },
    { title: "Stage 1 CNN", desc: "Diabetes Classifier", icon: Cpu, details: "Deep Convolutional network runs binary classification check." },
    { title: "Stage 2 CNN", desc: "Glucose Regressor", icon: Server, details: "Isolates Pancreas Sector ROI to estimate scalar blood sugar value." },
    { title: "Clarke Grid", desc: "Clinical Report", icon: BarChart2, details: "Validates predictions against clinical limits and zone boundaries." }
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
      setActiveNodeIndex(0);
      const timer = setInterval(() => {
        setActiveNodeIndex(prev => {
          if (prev < architectureNodes.length - 1) {
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
      setActiveNodeIndex(-1);
      setSelectedNodeDetails(null);
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
      setActiveNodeIndex(-1);
      setSelectedNodeDetails(null);
    }
  };

  const runAnalysis = async () => {
    if (!file) return;

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
    setSelectedNodeDetails(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Set a 10 second timeout for fetch
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 12000);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

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
      console.error(err);
      if (err.name === 'AbortError') {
        setError('Network timeout. Please check if your FastAPI backend server is running locally on port 8000.');
      } else {
        setError(err.message || 'Cannot connect to backend. Please ensure uvicorn is running at http://localhost:8000.');
      }
      setActiveNodeIndex(-1);
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
    setActiveNodeIndex(-1);
    setSelectedNodeDetails(null);
  };

  const getGlucoseColor = (val: number) => {
    if (val > 180) return '#dc2626'; 
    if (val > 120) return '#ea580c'; 
    return '#0d9488'; 
  };

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

  const getNodeClinicalValue = (idx: number) => {
    if (!prediction) return null;
    switch(idx) {
      case 0: return file?.name;
      case 1: return `Sharpness: ${prediction.sharpness.toFixed(1)} (PASS)`;
      case 2: return "Iris: Remapped (360° x 60)";
      case 3: return `Sigmoid: ${prediction.diagnosis} (${(prediction.diabetes_probability * 100).toFixed(0)}%)`;
      case 4: return prediction.diagnosis === 'Diabetic' ? `${prediction.estimated_glucose?.toFixed(0)} mg/dL` : "Bypassed";
      case 5: return prediction.diagnosis === 'Diabetic' ? "Zone A / Zone B Plot Ready" : "Normal control";
      default: return null;
    }
  };

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
        
        {/* Sleek Welcoming & System Scope Page (Shown when no image is uploaded) */}
        {!previewUrl && (
          <div className="glass animate-fade-in" style={{ padding: '36px', display: 'flex', flexDirection: 'column', gap: '30px' }}>
            
            {/* Header Welcome banner */}
            <div style={{ textAlign: 'center', borderBottom: '1px solid rgba(14, 165, 233, 0.1)', paddingBottom: '24px' }}>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: 'rgba(14,165,233,0.06)', padding: '6px 16px', borderRadius: '30px', border: '1px solid rgba(14,165,233,0.12)', color: '#0284c7', fontSize: '12px', fontWeight: 700, marginBottom: '16px' }}>
                <Activity size={14} /> AI-Powered Non-Invasive Diagnostics Portal
              </div>
              <h1 style={{ fontSize: '32px', fontWeight: 700, margin: '0 0 10px 0', color: '#0f172a', letterSpacing: '-0.75px' }}>
                Welcome to Iris DiaScan Workspace
              </h1>
              <p style={{ color: '#475569', fontSize: '15px', maxWidth: '640px', margin: '0 auto', lineHeight: 1.6 }}>
                Inspect topographical reflex zones of the human iris to classify diabetic markers and estimate glycemic trends non-invasively using cascaded neural networks.
              </p>
            </div>

            {/* Scope / Capabilities split panels */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
              
              {/* Card 1: What it does */}
              <div style={{ background: '#f8fafc', border: '1px solid rgba(15,23,42,0.05)', borderRadius: '12px', padding: '20px' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', fontWeight: 700, margin: '0 0 12px 0', color: '#0369a1' }}>
                  <Eye size={18} /> System Scope & Mechanics
                </h3>
                <ul style={{ margin: 0, paddingLeft: '18px', fontSize: '13px', color: '#475569', display: 'flex', flexDirection: 'column', gap: '8px', lineHeight: 1.5 }}>
                  <li><strong>Cascaded Gating</strong>: Splits diagnosis into binary detection (Stage 1) and value regression (Stage 2) for maximum efficiency.</li>
                  <li><strong>Optical Mapping</strong>: Emulates Daugman's model by unwarping circular irises into flat rectangular strips to align anatomical structures.</li>
                </ul>
              </div>

              {/* Card 2: Capabilities */}
              <div style={{ background: '#f0fdf4', border: '1px solid rgba(22,163,74,0.08)', borderRadius: '12px', padding: '20px' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', fontWeight: 700, margin: '0 0 12px 0', color: '#15803d' }}>
                  <CheckCircle size={18} /> System Capabilities (CAN DO)
                </h3>
                <ul style={{ margin: 0, paddingLeft: '18px', fontSize: '13px', color: '#3f6212', display: 'flex', flexDirection: 'column', gap: '8px', lineHeight: 1.5 }}>
                  <li>Locate boundaries of the pupil and outer limbus automatically.</li>
                  <li>Check image sharpness variance to reject blurry/unfocused scans.</li>
                  <li>Provide real-time interactive Clarke Error Grid plots for glucose levels.</li>
                  <li>Simulate tissue variations in the reflex zone (Digital Twin).</li>
                </ul>
              </div>

              {/* Card 3: Limitations */}
              <div style={{ background: '#fff7ed', border: '1px solid rgba(234,88,12,0.08)', borderRadius: '12px', padding: '20px' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', fontWeight: 700, margin: '0 0 12px 0', color: '#c2410c' }}>
                  <AlertTriangle size={18} /> Clinical Boundaries (CANNOT DO)
                </h3>
                <ul style={{ margin: 0, paddingLeft: '18px', fontSize: '13px', color: '#7c2d12', display: 'flex', flexDirection: 'column', gap: '8px', lineHeight: 1.5 }}>
                  <li>Does NOT replace professional clinical blood draws or CGM trackers.</li>
                  <li>Cannot process low-resolution or dark/occluded iris photography.</li>
                  <li>Stage 2 Regressor is a mock blueprint (requires clinical calibration dataset).</li>
                </ul>
              </div>

            </div>

            {/* Interactive Dropzone Uploader */}
            <div 
              className="upload-zone" 
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{ padding: '50px 30px', border: '2px dashed rgba(14, 165, 233, 0.2)', borderRadius: '12px', background: 'rgba(255,255,255,0.4)', textAlign: 'center', cursor: 'pointer', transition: 'all 0.3s' }}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                style={{ display: 'none' }}
                accept="image/*"
              />
              <Upload size={38} color="#0ea5e9" style={{ marginBottom: '12px', filter: 'drop-shadow(0 2px 6px rgba(14, 165, 233, 0.2))' }} />
              <h4 style={{ fontSize: '18px', fontWeight: 600, margin: '0 0 6px 0', color: '#0f172a' }}>
                Load Patient Iris Scan
              </h4>
              <p style={{ color: '#64748b', fontSize: '13px', margin: 0 }}>
                Click to browse local folders or drag-and-drop a close-up photo of the eye.
              </p>
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
              
              <div style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', border: '1px solid rgba(14, 165, 233, 0.1)', background: '#f1f5f9', aspectRatio: '1/1' }}>
                <img 
                  src={previewUrl} 
                  alt="Patient Scan" 
                  style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
                />
                
                {prediction && (
                  <svg 
                    viewBox="0 0 300 300" 
                    style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                  >
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
                    
                    <path 
                      d="M 150 118 L 150 58 A 92 92 0 0 1 215 85 L 172.6 127.4 A 32 32 0 0 0 150 118 Z" 
                      fill={twinStyle.fill} 
                      stroke={twinStyle.stroke} 
                      strokeWidth="1.5" 
                      style={{ transition: 'all 0.4s ease' }}
                    />

                    <text x="175" y="70" fill={twinStyle.stroke} fontSize="8.5" fontWeight="bold" opacity="0.9">PANCREAS ROI</text>
                    <line x1="150" y1="150" x2="220" y2="150" stroke="rgba(15, 23, 42, 0.1)" strokeWidth="0.5" strokeDasharray="2 2" />
                    <line x1="150" y1="150" x2="150" y2="50" stroke="rgba(15, 23, 42, 0.1)" strokeWidth="0.5" strokeDasharray="2 2" />
                  </svg>
                )}
              </div>

              {error && (
                <div style={{ background: 'rgba(220, 38, 38, 0.05)', border: '1px solid rgba(220, 38, 38, 0.15)', color: '#dc2626', borderRadius: '8px', padding: '12px', fontSize: '13px', display: 'flex', gap: '8px', alignItems: 'flex-start', fontWeight: 500 }}>
                  <AlertCircle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
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
                  {loading ? (
                    <>
                      <RefreshCw size={18} className="animate-spin" style={{ animation: 'spin 1.5s linear infinite' }} />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <Activity size={18} />
                      <span>Run Diagnostics</span>
                    </>
                  )}
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

            {/* Right Box: Live Pipeline Diagram Panel */}
            <div className="glass" style={{ padding: '24px', display: 'flex', flexDirection: 'column', minHeight: '400px', justifyContent: 'space-between' }}>
              
              <div>
                <h3 style={{ fontSize: '16px', fontWeight: 600, margin: '0 0 4px 0', color: '#0f172a', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Server size={18} color="#0ea5e9" /> Interactive Pipeline Architecture
                </h3>
                <p style={{ margin: '0 0 16px 0', fontSize: '12px', color: '#64748b' }}>
                  Click on completed nodes to review exact inputs/outputs processing logs.
                </p>

                {/* Pipeline Node Map */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', position: 'relative' }}>
                  {architectureNodes.map((node, idx) => {
                    const isNodeCompleted = prediction || (loading && idx < activeNodeIndex);
                    const isNodeActive = loading && idx === activeNodeIndex;
                    const clinicalVal = getNodeClinicalValue(idx);
                    
                    const NodeIcon = node.icon;

                    return (
                      <div 
                        key={idx}
                        onClick={() => isNodeCompleted && setSelectedNodeDetails(node.details + (clinicalVal ? `\n\nProcessed values: ${clinicalVal}` : ''))}
                        style={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: '12px', 
                          padding: '10px 14px', 
                          borderRadius: '8px',
                          border: isNodeActive ? '1px solid #0ea5e9' : '1px solid rgba(15,23,42,0.06)',
                          background: isNodeActive ? 'rgba(14, 165, 233, 0.04)' : (isNodeCompleted ? '#f8fafc' : 'rgba(15,23,42,0.01)'),
                          cursor: isNodeCompleted ? 'pointer' : 'default',
                          transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
                          boxShadow: isNodeActive ? '0 0 8px rgba(14, 165, 233, 0.25)' : 'none',
                          transform: isNodeActive ? 'translateX(4px)' : 'none'
                        }}
                      >
                        <div style={{ 
                          width: '32px', 
                          height: '32px', 
                          borderRadius: '50%', 
                          background: isNodeActive ? '#0ea5e9' : (isNodeCompleted ? '#e2fbf4' : '#f1f5f9'),
                          display: 'flex', 
                          justifyContent: 'center', 
                          alignItems: 'center',
                          color: isNodeActive ? '#ffffff' : (isNodeCompleted ? '#0d9488' : '#94a3b8')
                        }}>
                          {isNodeCompleted ? <CheckCircle size={18} /> : <NodeIcon size={16} />}
                        </div>

                        <div style={{ flex: 1 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ fontSize: '13px', fontWeight: 600, color: isNodeActive ? '#0ea5e9' : '#0f172a' }}>{node.title}</span>
                            {clinicalVal && <span style={{ fontSize: '11px', color: '#0d9488', fontWeight: 700 }}>{clinicalVal}</span>}
                          </div>
                          <span style={{ fontSize: '11px', color: '#64748b' }}>{node.desc}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Node Details Overlay Panel */}
              {selectedNodeDetails ? (
                <div style={{ 
                  marginTop: '16px', 
                  background: '#f0f9ff', 
                  border: '1px solid #bae6fd', 
                  borderRadius: '8px', 
                  padding: '12px',
                  fontSize: '12.5px',
                  color: '#0369a1',
                  animation: 'fadeIn 0.2s ease-out'
                }}>
                  <div style={{ fontWeight: 700, marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}><HelpCircle size={14} /> Node Inspection Log:</div>
                  <div style={{ fontStyle: 'italic', color: '#0284c7', whiteSpace: 'pre-line' }}>{selectedNodeDetails}</div>
                </div>
              ) : (
                prediction && (
                  <div style={{ fontSize: '11.5px', color: '#64748b', textAlign: 'center', marginTop: '14px' }}>
                    💡 Tap any step above to inspect its clinical diagnostic parameter logs.
                  </div>
                )
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
