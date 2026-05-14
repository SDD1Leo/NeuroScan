import React, { useState, useRef, useEffect } from 'react';
import { 
  UploadCloud, 
  FileImage, 
  Activity, 
  Brain, 
  Clock, 
  Maximize, 
  RefreshCw, 
  AlertCircle,
  ShieldCheck,
  AlertTriangle,
  Cpu,
  Layers,
  Database,
  Info,
  Sparkles,
  MessageCircle,
  X,
  Send,
  User,
  Bot
} from 'lucide-react';

// --- Utility: Exponential Backoff Fetch ---
const fetchWithRetry = async (url, options, retries = 5) => {
  const delays = [1000, 2000, 4000, 8000, 16000];
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url, options);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return response;
    } catch (e) {
      if (i === retries - 1) throw e;
      await new Promise(res => setTimeout(res, delays[i]));
    }
  }
};

// --- Utility: File to Base64 ---
const getBase64 = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = () => resolve(reader.result);
  reader.onerror = error => reject(error);
});

export default function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Gemini AI Report States
  const [isGeneratingAI, setIsGeneratingAI] = useState(false);
  const [aiReport, setAiReport] = useState(null);
  const [aiError, setAiError] = useState(null);
  
  // Chatbot States
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    { role: 'model', text: "Hello! I'm the NeuroScan AI assistant. How can I help you understand your brain health or MRI results today?" }
  ]);
  
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  // The endpoint URL provided
  const API_ENDPOINT = 'https://newcastle-viruses-bool-characteristics.trycloudflare.com/predict';

  // Cleanup object URL on unmount
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (isChatOpen) {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages, isChatOpen, isChatLoading]);

  const handleFileSelect = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please select a valid image file (JPEG, PNG).');
      return;
    }
    
    setError(null);
    setResult(null);
    setAiReport(null);
    setAiError(null);
    setSelectedFile(file);
    
    // Create preview URL
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setAiReport(null);
    setAiError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'accept': 'application/json',
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('API Error:', err);
      setError('Failed to connect to the endpoint. The Cloudflare tunnel might have expired or CORS is preventing the request.');
    } finally {
      setIsLoading(false);
    }
  };

  // --- Gemini API: Clinical Insights ---
  const generateAIInsights = async () => {
    if (!selectedFile || !result) return;
    
    setIsGeneratingAI(true);
    setAiError(null);

    try {
      const apiKey = "AIzaSyA2TRCu-XLV9imiRLYmuY6rWHLwvhxn21g"; 
      const base64String = await getBase64(selectedFile);
      const base64Data = base64String.split(',')[1];
      
      let mimeType = selectedFile.type;
      if (mimeType === 'image/jpg') mimeType = 'image/jpeg';
      if (!mimeType) mimeType = 'image/jpeg';
      
      const prompt = `System Instruction: You are an expert neurological AI assistant. Provide educational information based on MRI scans and model classifications. You are not providing a medical diagnosis.

      Our primary CNN model has classified this brain MRI scan as "${result.prediction}" with ${result.confidence.toFixed(1)}% confidence.
      
      Please provide a structured, educational report including:
      1. A brief explanation of what "${result.prediction}" means in the context of brain MRI scans.
      2. General visual characteristics typically associated with this classification.
      3. Common symptoms a patient might experience.
      4. Standard next steps a doctor might take for diagnosis or verification.
      
      Keep it professional, empathetic, and strictly educational. Conclude with a clear disclaimer that this is an AI educational tool and not medical advice. Use simple formatting.`;

      const payload = {
        contents: [{
          role: "user",
          parts: [
            { text: prompt },
            { inlineData: { mimeType: mimeType, data: base64Data } }
          ]
        }]
      };

      const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;      
      const response = await fetchWithRetry(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      if (data.candidates && data.candidates[0]) {
         setAiReport(data.candidates[0].content.parts[0].text);
      } else {
         throw new Error("Invalid response structure from Gemini API");
      }
    } catch (err) {
      console.error('Gemini API Error:', err);
      setAiError("Failed to generate AI insights. The service might be temporarily unavailable.");
    } finally {
      setIsGeneratingAI(false);
    }
  };

  // --- Gemini API: Chatbot ---
  const handleSendMessage = async (e) => {
    e?.preventDefault();
    if (!chatInput.trim() || isChatLoading) return;

    const userMessage = chatInput.trim();
    setChatInput('');
    
    // Add user message to UI immediately
    const updatedMessages = [...chatMessages, { role: 'user', text: userMessage }];
    setChatMessages(updatedMessages);
    setIsChatLoading(true);

    try {
      const apiKey = "AIzaSyA2TRCu-XLV9imiRLYmuY6rWHLwvhxn21g";
      const endpoint =`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;
      // Build the conversation history
      const historyContents = updatedMessages.map(msg => ({
        role: msg.role === 'model' ? 'model' : 'user',
        parts: [{ text: msg.text }]
      }));

      // To ensure the proxy accepts it safely, we append the system context to the latest user message
      // instead of using a separate systemInstruction block.
      const lastMessageIndex = historyContents.length - 1;
      const contextPrefix = `[System Context: You are a compassionate, knowledgeable medical AI assistant for the NeuroScan app. Answer patient questions. Always remind them to consult a real doctor for medical advice. Keep answers concise, readable, and comforting. ${result ? `The patient using this app recently had an MRI scan evaluated by our CNN, which predicted "${result.prediction}" with ${result.confidence.toFixed(1)}% confidence.` : 'The patient has not uploaded a scan yet.'}]\n\nPatient Query: `;
      
      historyContents[lastMessageIndex].parts[0].text = contextPrefix + userMessage;

      const payload = {
        contents: historyContents
      };

      const response = await fetchWithRetry(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      if (data.candidates && data.candidates[0]) {
        const botReply = data.candidates[0].content.parts[0].text;
        setChatMessages(prev => [...prev, { role: 'model', text: botReply }]);
      } else {
        throw new Error("Invalid chat response");
      }
    } catch (error) {
      console.error('Chat API Error:', error);
      setChatMessages(prev => [...prev, { role: 'model', text: "I'm sorry, I'm having trouble connecting to the network right now. Please try asking again in a moment." }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setAiReport(null);
    setAiError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const loadMockData = () => {
    setIsLoading(true);
    setError(null);
    setTimeout(() => {
      setResult({
        "prediction": "Pituitary",
        "confidence": 91.86009764671326,
        "class_probabilities": {
          "Pituitary": 91.86009979248047,
          "Glioma": 4.565362453460693,
          "No Tumor": 3.541550874710083,
          "Meningioma": 0.03299401327967644
        },
        "inference_time": 0.08,
        "is_tumor": true,
        "input_size": "128x128"
      });
      setIsLoading(false);
    }, 1200);
  };

  // Helper to nicely format markdown response from Gemini
  const formatMarkdown = (text) => {
    return text.split('\n').map((line, i) => {
      if (!line.trim()) return <div key={i} className="h-1"></div>;
      let formattedLine = line;
      formattedLine = formattedLine.replace(/\*\*(.*?)\*\*/g, '<strong class="text-gray-100">$1</strong>');
      formattedLine = formattedLine.replace(/\*(.*?)\*/g, '<em class="text-gray-300">$1</em>');
      
      if (formattedLine.startsWith('## ') || formattedLine.startsWith('### ')) {
        return <h4 key={i} className="text-sm font-bold text-teal-300 mt-2 mb-1" dangerouslySetInnerHTML={{ __html: formattedLine.replace(/#+\s/, '') }} />;
      }
      if (formattedLine.startsWith('* ') || formattedLine.startsWith('- ')) {
        return (
          <div key={i} className="flex gap-2 mb-1 pl-1">
            <span className="text-teal-500 mt-0.5">•</span>
            <span dangerouslySetInnerHTML={{ __html: formattedLine.substring(2) }} />
          </div>
        );
      }
      return <p key={i} className="mb-1 leading-relaxed" dangerouslySetInnerHTML={{ __html: formattedLine }} />;
    });
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-sans relative">
      {/* Navigation Bar */}
      <nav className="bg-gray-900 border-b border-gray-800 shadow-sm sticky top-0 z-20">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div 
            className="flex items-center gap-3 cursor-pointer group"
            onClick={() => setCurrentPage('home')}
          >
            <div className="bg-teal-600 text-white p-2 rounded-lg group-hover:bg-teal-500 transition-colors">
              <Brain size={28} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-100 leading-tight group-hover:text-white transition-colors">NeuroScan AI</h1>
              <p className="text-xs text-teal-400/80 font-medium tracking-wide uppercase">Tumor Classification</p>
            </div>
          </div>
          <div className="flex gap-2">
            <button 
              onClick={() => setCurrentPage('home')}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition-all ${
                currentPage === 'home' 
                  ? 'bg-teal-900/40 text-teal-400 border border-teal-800/50' 
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800 border border-transparent'
              }`}
            >
              Scanner
            </button>
            <button 
              onClick={() => setCurrentPage('model')}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition-all ${
                currentPage === 'model' 
                  ? 'bg-teal-900/40 text-teal-400 border border-teal-800/50' 
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800 border border-transparent'
              }`}
            >
              Our Model
            </button>
          </div>
        </div>
      </nav>

      {/* Main Scanner Content */}
      {currentPage === 'home' && (
        <main className="max-w-6xl mx-auto px-4 py-8 animate-in fade-in duration-300">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            
            {/* Left Column: Upload & Preview */}
            <div className="space-y-6">
              <div className="bg-gray-900 rounded-2xl p-6 shadow-sm border border-gray-800">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <FileImage className="text-teal-400" size={20} />
                  Input Image
                </h2>

                {!selectedFile ? (
                  <div
                    className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center text-center transition-colors ${
                      isDragging ? 'border-teal-500 bg-teal-900/20' : 'border-gray-700 hover:border-teal-500 hover:bg-gray-800/50'
                    } cursor-pointer min-h-[300px]`}
                    onDragOver={onDragOver}
                    onDragLeave={onDragLeave}
                    onDrop={onDrop}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input
                      type="file"
                      ref={fileInputRef}
                      className="hidden"
                      accept="image/*"
                      onChange={(e) => handleFileSelect(e.target.files?.[0])}
                    />
                    <div className="bg-teal-900/40 text-teal-400 p-4 rounded-full mb-4">
                      <UploadCloud size={32} />
                    </div>
                    <p className="text-gray-300 font-medium mb-1">Click or drag image to upload</p>
                    <p className="text-gray-500 text-sm">Supports JPG, PNG (MRI Scans)</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="relative rounded-xl overflow-hidden bg-black aspect-square max-h-[400px] flex items-center justify-center group border border-gray-800">
                      <img 
                        src={previewUrl} 
                        alt="MRI Scan Preview" 
                        className="max-w-full max-h-full object-contain"
                      />
                      <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <button 
                          onClick={() => fileInputRef.current?.click()}
                          className="bg-gray-800 text-gray-200 border border-gray-700 px-4 py-2 rounded-lg font-medium text-sm hover:bg-gray-700 transition-colors flex items-center gap-2 shadow-lg"
                        >
                          <RefreshCw size={16} />
                          Change Image
                        </button>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between bg-gray-800 p-3 rounded-lg border border-gray-700">
                      <div className="flex items-center gap-2 overflow-hidden">
                        <FileImage size={18} className="text-gray-500 flex-shrink-0" />
                        <span className="text-sm text-gray-200 truncate font-medium">
                          {selectedFile.name}
                        </span>
                      </div>
                      <span className="text-xs text-gray-400 font-medium">
                        {(selectedFile.size / 1024).toFixed(1)} KB
                      </span>
                    </div>

                    {!result && !isLoading && (
                      <button
                        onClick={handleAnalyze}
                        className="w-full bg-teal-600 hover:bg-teal-500 text-white font-semibold py-3 px-4 rounded-xl shadow-md shadow-teal-900/30 transition-all active:scale-[0.98] flex justify-center items-center gap-2"
                      >
                        <Activity size={20} />
                        Analyze MRI Scan
                      </button>
                    )}
                  </div>
                )}
              </div>
              
              {/* Error Message & Mock Data Fallback */}
              {error && (
                <div className="bg-red-900/20 border border-red-800/50 rounded-xl p-4 flex flex-col gap-3 text-red-400 text-sm">
                  <div className="flex gap-3">
                    <AlertCircle className="flex-shrink-0 mt-0.5" size={18} />
                    <p>{error}</p>
                  </div>
                  {/* Fallback button for UI preview if API is down */}
                  <button 
                    onClick={loadMockData}
                    className="self-start bg-red-900/40 hover:bg-red-900/60 text-red-300 font-medium py-1.5 px-3 rounded-lg text-xs transition-colors border border-red-800/50"
                  >
                    Load Mock Data (Test UI)
                  </button>
                </div>
              )}
            </div>

            {/* Right Column: Results & Gemini AI Report */}
            <div className="space-y-6">
              <div className="bg-gray-900 rounded-2xl p-6 shadow-sm border border-gray-800 h-full flex flex-col relative overflow-hidden">
                <h2 className="text-lg font-semibold mb-6 flex items-center gap-2 text-gray-100">
                  <Activity className="text-teal-400" size={20} />
                  Analysis Results
                </h2>

                {isLoading ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-gray-500 min-h-[300px]">
                    <div className="relative">
                          <div className="w-16 h-16 border-4 border-gray-800 rounded-full"></div>
                          <div className="w-16 h-16 border-4 border-teal-500 rounded-full border-t-transparent animate-spin absolute top-0 left-0"></div>
                    </div>
                    <p className="mt-4 font-medium text-gray-400 animate-pulse">Running neural network inference...</p>
                    <p className="text-xs text-gray-600 mt-1">Analyzing scan at {API_ENDPOINT.split('.')[0].split('//')[1]}...</p>
                  </div>
                ) : result ? (
                  <div className="flex-1 space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    
                    {/* Primary Diagnosis Card */}
                    <div className={`p-5 rounded-xl border ${result.is_tumor ? 'bg-orange-950/30 border-orange-900/50' : 'bg-teal-950/30 border-teal-900/50'} relative overflow-hidden`}>
                      <div className="flex items-start justify-between">
                        <div>
                          <p className={`text-sm font-semibold mb-1 ${result.is_tumor ? 'text-orange-500' : 'text-teal-500'}`}>
                            Primary Classification
                          </p>
                          <h3 className={`text-3xl font-bold ${result.is_tumor ? 'text-orange-400' : 'text-teal-400'}`}>
                            {result.prediction}
                          </h3>
                          {result.is_tumor ? (
                            <div className="flex items-center gap-1.5 mt-2 text-orange-400/80 text-sm font-medium">
                              <AlertTriangle size={16} /> Tumor markers detected
                            </div>
                          ) : (
                            <div className="flex items-center gap-1.5 mt-2 text-teal-400/80 text-sm font-medium">
                              <ShieldCheck size={16} /> No distinct tumor markers
                            </div>
                          )}
                        </div>
                        <div className={`text-right p-3 rounded-lg ${result.is_tumor ? 'bg-orange-900/40 text-orange-400' : 'bg-teal-900/40 text-teal-400'}`}>
                          <div className="text-2xl font-bold leading-none mb-1">
                            {result.confidence.toFixed(1)}%
                          </div>
                          <div className="text-[10px] uppercase tracking-wider font-semibold opacity-80">
                            Confidence
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Probabilities Breakdown */}
                    <div>
                      <h4 className="text-sm font-semibold text-gray-300 mb-3">Class Probabilities</h4>
                      <div className="space-y-3">
                        {Object.entries(result.class_probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .map(([className, probability]) => {
                            const isPrimary = className === result.prediction;
                            const percent = probability.toFixed(2);
                            
                            return (
                              <div key={className} className="space-y-1">
                                <div className="flex justify-between text-sm">
                                  <span className={`font-medium ${isPrimary ? 'text-gray-100' : 'text-gray-400'}`}>
                                    {className}
                                  </span>
                                  <span className={isPrimary ? 'font-bold text-gray-100' : 'text-gray-500'}>
                                    {percent}%
                                  </span>
                                </div>
                                <div className="h-2.5 w-full bg-gray-800 rounded-full overflow-hidden">
                                  <div 
                                    className={`h-full rounded-full transition-all duration-1000 ease-out ${
                                      isPrimary 
                                        ? (result.is_tumor ? 'bg-orange-500' : 'bg-teal-500') 
                                        : 'bg-gray-600'
                                    }`}
                                    style={{ width: `${Math.max(probability, 0.5)}%` }}
                                  ></div>
                                </div>
                              </div>
                            );
                          })}
                      </div>
                    </div>

                    {/* Metadata & Gemini Integration Section */}
                    <div className="border-t border-gray-800 pt-6">
                      
                      {!aiReport && !isGeneratingAI ? (
                        <div className="bg-gradient-to-br from-indigo-900/40 to-purple-900/40 border border-indigo-800/50 rounded-xl p-5 shadow-inner">
                          <div className="flex items-start gap-4">
                            <div className="bg-indigo-500/20 p-2.5 rounded-lg text-indigo-400">
                              <Sparkles size={24} />
                            </div>
                            <div>
                              <h4 className="text-white font-semibold mb-1">Need more context?</h4>
                              <p className="text-indigo-200/80 text-sm mb-4 leading-relaxed">
                                Use Gemini AI to analyze this classification and generate a detailed educational report on {result.prediction.toLowerCase()} characteristics and standard procedures.
                              </p>
                              <button
                                onClick={generateAIInsights}
                                className="bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center gap-2 text-sm shadow-md shadow-indigo-900/20"
                              >
                                Generate AI Report ✨
                              </button>
                            </div>
                          </div>
                        </div>
                      ) : isGeneratingAI ? (
                         <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 flex flex-col items-center justify-center gap-3">
                           <Sparkles className="text-indigo-400 animate-pulse" size={28} />
                           <p className="text-gray-300 font-medium animate-pulse">Gemini is analyzing the scan...</p>
                         </div>
                      ) : (
                         <div className="bg-gray-800/40 border border-indigo-900/40 rounded-xl overflow-hidden shadow-inner">
                           <div className="bg-indigo-900/30 border-b border-indigo-900/40 p-4 flex items-center justify-between">
                             <h4 className="text-indigo-300 font-semibold flex items-center gap-2">
                               <Sparkles size={18} />
                               AI Clinical Insights
                             </h4>
                             <span className="text-xs bg-indigo-900/50 text-indigo-300 px-2 py-1 rounded font-medium border border-indigo-800/50">
                               Powered by Gemini
                             </span>
                           </div>
                           <div className="p-5 text-sm text-gray-300 overflow-y-auto max-h-[400px] custom-scrollbar">
                             {formatMarkdown(aiReport)}
                           </div>
                         </div>
                      )}

                      {aiError && (
                        <div className="mt-3 text-red-400 text-sm flex gap-2 items-center bg-red-900/20 p-3 rounded-lg border border-red-800/30">
                          <AlertCircle size={16} /> {aiError}
                        </div>
                      )}
                    </div>

                    <button
                      onClick={handleReset}
                      className="w-full mt-2 bg-gray-900 border-2 border-gray-700 hover:border-teal-500 hover:bg-gray-800 text-gray-300 font-semibold py-2.5 px-4 rounded-xl transition-all active:scale-[0.98]"
                    >
                      Analyze Another Image
                    </button>

                  </div>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-gray-500 min-h-[300px]">
                    <Brain size={48} className="text-gray-800 mb-4" strokeWidth={1} />
                    <p className="text-center max-w-[250px] text-gray-400">Upload an MRI scan and click analyze to view the AI prediction results here.</p>
                  </div>
                )}
              </div>
            </div>

          </div>
        </main>
      )}

      {/* Model Explanation Page */}
      {currentPage === 'model' && (
        <main className="max-w-6xl mx-auto px-4 py-8 animate-in fade-in duration-500">
          <div className="mb-10 text-center max-w-3xl mx-auto mt-4">
            <h2 className="text-3xl font-bold text-gray-100 mb-4">Behind the AI: Our CNN Architecture</h2>
            <p className="text-gray-400 text-lg leading-relaxed">
              NeuroScan AI is powered by a state-of-the-art Convolutional Neural Network (CNN) specifically trained to identify and classify brain tumors from MRI scans with high diagnostic accuracy.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-gray-900 border border-gray-800 p-6 rounded-2xl shadow-sm hover:border-teal-500/50 transition-colors group">
              <div className="bg-teal-900/30 w-12 h-12 rounded-xl flex items-center justify-center mb-5 text-teal-400 group-hover:scale-110 transition-transform">
                <Database size={24} />
              </div>
              <h3 className="text-xl font-semibold text-gray-100 mb-3">Training Data</h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                Trained on thousands of curated MRI scans featuring 4 distinct classes: Glioma, Meningioma, Pituitary tumors, and healthy brains. Data augmentation techniques like rotation and flipping ensure the model's robustness.
              </p>
            </div>

            <div className="bg-gray-900 border border-gray-800 p-6 rounded-2xl shadow-sm hover:border-teal-500/50 transition-colors group">
              <div className="bg-teal-900/30 w-12 h-12 rounded-xl flex items-center justify-center mb-5 text-teal-400 group-hover:scale-110 transition-transform">
                <Layers size={24} />
              </div>
              <h3 className="text-xl font-semibold text-gray-100 mb-3">Deep Layers</h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                Utilizes deep convolutional layers to extract hierarchical features. Early layers detect basic edges and shapes, while deeper layers identify complex tumor-specific textures and boundaries within the brain tissue.
              </p>
            </div>

            <div className="bg-gray-900 border border-gray-800 p-6 rounded-2xl shadow-sm hover:border-teal-500/50 transition-colors group">
              <div className="bg-teal-900/30 w-12 h-12 rounded-xl flex items-center justify-center mb-5 text-teal-400 group-hover:scale-110 transition-transform">
                <Cpu size={24} />
              </div>
              <h3 className="text-xl font-semibold text-gray-100 mb-3">Fast Inference</h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                Optimized for rapid processing. Input images are standardized to 128x128 pixels, allowing the neural network to execute feed-forward inference in a fraction of a second without sacrificing accuracy.
              </p>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8 shadow-sm">
            <div className="flex items-center gap-3 mb-8">
              <Info className="text-teal-400" size={24} />
              <h3 className="text-2xl font-semibold text-gray-100">Network Pipeline</h3>
            </div>
            
            {/* Visual Pipeline */}
            <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
              
              <div className="flex-1 bg-gray-950 border border-gray-800 p-5 rounded-xl text-center relative shadow-inner">
                <div className="text-teal-400 font-bold mb-2">1. Input</div>
                <div className="text-sm text-gray-400">MRI Image<br/>(128x128x3)</div>
                <div className="hidden md:block absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-0.5 bg-gray-700"></div>
                <div className="md:hidden absolute -bottom-3 left-1/2 -translate-x-1/2 h-6 w-0.5 bg-gray-700"></div>
              </div>

              <div className="flex-1 bg-gray-950 border border-gray-800 p-5 rounded-xl text-center relative shadow-inner">
                <div className="text-teal-400 font-bold mb-2">2. Conv Blocks</div>
                <div className="text-sm text-gray-400">Conv2D + ReLU<br/>+ MaxPool</div>
                <div className="hidden md:block absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-0.5 bg-gray-700"></div>
                <div className="md:hidden absolute -bottom-3 left-1/2 -translate-x-1/2 h-6 w-0.5 bg-gray-700"></div>
              </div>

              <div className="flex-1 bg-gray-950 border border-gray-800 p-5 rounded-xl text-center relative shadow-inner">
                <div className="text-teal-400 font-bold mb-2">3. Dense Layers</div>
                <div className="text-sm text-gray-400">Flatten + FC<br/>+ Dropout</div>
                <div className="hidden md:block absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-0.5 bg-gray-700"></div>
                <div className="md:hidden absolute -bottom-3 left-1/2 -translate-x-1/2 h-6 w-0.5 bg-gray-700"></div>
              </div>

              <div className="flex-1 bg-gray-950 border border-gray-800 p-5 rounded-xl text-center shadow-inner">
                <div className="text-teal-400 font-bold mb-2">4. Output</div>
                <div className="text-sm text-gray-400">Softmax<br/>(4 Classes)</div>
              </div>

            </div>
          </div>
        </main>
      )}

      {/* --- Floating Chatbot UI --- */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end">
        {/* Chat Window */}
        <div 
          className={`transition-all duration-300 transform origin-bottom-right ${
            isChatOpen ? 'scale-100 opacity-100 mb-4' : 'scale-0 opacity-0 h-0 w-0 overflow-hidden'
          }`}
        >
          <div className="w-80 sm:w-96 bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl flex flex-col overflow-hidden h-[500px] max-h-[80vh]">
            
            {/* Header */}
            <div className="bg-teal-900/60 p-4 border-b border-teal-800/50 flex justify-between items-center">
              <div className="flex items-center gap-2">
                <div className="bg-teal-500 p-1.5 rounded-lg text-white">
                  <Bot size={20} />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-100 text-sm">Neuro Assistant</h3>
                  <div className="flex items-center gap-1 text-[10px] text-teal-300">
                    <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse"></span>
                    Online
                  </div>
                </div>
              </div>
              <button 
                onClick={() => setIsChatOpen(false)}
                className="text-teal-300 hover:text-white hover:bg-teal-800/50 p-1.5 rounded-lg transition-colors"
              >
                <X size={18} />
              </button>
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar bg-gray-950/50">
              {chatMessages.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`flex gap-2 max-w-[85%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                    
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1 ${
                      msg.role === 'user' ? 'bg-gray-800 text-gray-300 border border-gray-700' : 'bg-teal-900/50 text-teal-400 border border-teal-800/50'
                    }`}>
                      {msg.role === 'user' ? <User size={16} /> : <Sparkles size={16} />}
                    </div>

                    <div className={`p-3 rounded-2xl text-sm shadow-sm ${
                      msg.role === 'user' 
                        ? 'bg-gray-800 text-gray-100 border border-gray-700 rounded-tr-none' 
                        : 'bg-teal-950/40 text-gray-200 border border-teal-900/50 rounded-tl-none'
                    }`}>
                      {msg.role === 'model' ? (
                        <div className="space-y-1">
                          {formatMarkdown(msg.text)}
                        </div>
                      ) : (
                        msg.text
                      )}
                    </div>

                  </div>
                </div>
              ))}

              {isChatLoading && (
                <div className="flex justify-start">
                  <div className="flex gap-2 max-w-[85%] flex-row">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1 bg-teal-900/50 text-teal-400 border border-teal-800/50">
                      <Sparkles size={16} />
                    </div>
                    <div className="p-4 rounded-2xl text-sm shadow-sm bg-teal-950/40 border border-teal-900/50 rounded-tl-none flex gap-1 items-center h-10">
                      <span className="w-1.5 h-1.5 bg-teal-500 rounded-full animate-bounce"></span>
                      <span className="w-1.5 h-1.5 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                      <span className="w-1.5 h-1.5 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-3 bg-gray-900 border-t border-gray-800">
              <form 
                onSubmit={handleSendMessage}
                className="flex items-center gap-2 bg-gray-950 border border-gray-800 rounded-xl p-1 pr-2 shadow-inner"
              >
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask a medical question..."
                  className="flex-1 bg-transparent border-none focus:outline-none text-sm text-gray-100 px-3 py-2"
                  disabled={isChatLoading}
                />
                <button
                  type="submit"
                  disabled={!chatInput.trim() || isChatLoading}
                  className="bg-teal-600 hover:bg-teal-500 disabled:bg-gray-800 disabled:text-gray-500 text-white p-2 rounded-lg transition-colors flex-shrink-0"
                >
                  <Send size={16} className={isChatLoading ? "opacity-50" : ""} />
                </button>
              </form>
              <p className="text-center text-[9px] text-gray-500 mt-2 font-medium">
                AI can make mistakes. Consult your doctor for medical advice.
              </p>
            </div>
          </div>
        </div>

        {/* Floating Toggle Button */}
        <button
          onClick={() => setIsChatOpen(!isChatOpen)}
          className={`p-4 rounded-full shadow-lg shadow-teal-900/20 transition-all transform hover:scale-105 active:scale-95 flex items-center justify-center ${
            isChatOpen ? 'bg-gray-800 text-gray-400 hover:bg-gray-700' : 'bg-teal-600 text-white hover:bg-teal-500'
          }`}
        >
          {isChatOpen ? <X size={24} /> : <MessageCircle size={24} />}
        </button>
      </div>
      
      {/* Global CSS for scrollbar styling on the AI report & chat box */}
      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background-color: #4f46e5;
          border-radius: 10px;
        }
      `}} />
    </div>
  );
}