/**
 * @file demo.js
 * @brief Demo application for OdinFold++ WASM
 */

class OdinFoldDemo {
    constructor() {
        this.folder = null;
        this.currentResult = null;
        this.startTime = null;
        this.peakMemory = 0;
        
        this.initializeElements();
        this.setupEventListeners();
        this.updateSystemInfo();
        this.initializeWASM();
    }
    
    initializeElements() {
        // Status elements
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.status = document.getElementById('status');
        
        // Input elements
        this.sequenceInput = document.getElementById('sequenceInput');
        this.sequenceLength = document.getElementById('sequenceLength');
        this.sequenceValid = document.getElementById('sequenceValid');
        this.foldButton = document.getElementById('foldButton');
        
        // Progress elements
        this.progressSection = document.getElementById('progressSection');
        this.progressFill = document.getElementById('progressFill');
        this.progressPercent = document.getElementById('progressPercent');
        this.progressStatus = document.getElementById('progressStatus');
        this.memoryUsage = document.getElementById('memoryUsage');
        this.elapsedTime = document.getElementById('elapsedTime');
        
        // Results elements
        this.resultsSection = document.getElementById('resultsSection');
        this.resultLength = document.getElementById('resultLength');
        this.resultConfidence = document.getElementById('resultConfidence');
        this.resultTime = document.getElementById('resultTime');
        this.resultMemory = document.getElementById('resultMemory');
        this.confidenceChart = document.getElementById('confidenceChart');
        
        // Download elements
        this.downloadPDB = document.getElementById('downloadPDB');
        this.downloadJSON = document.getElementById('downloadJSON');
        
        // Error modal
        this.errorModal = document.getElementById('errorModal');
        this.errorMessage = document.getElementById('errorMessage');
        this.closeError = document.getElementById('closeError');
        this.errorOk = document.getElementById('errorOk');
    }
    
    setupEventListeners() {
        // Sequence input
        this.sequenceInput.addEventListener('input', () => this.validateSequence());
        this.sequenceInput.addEventListener('paste', () => {
            setTimeout(() => this.validateSequence(), 10);
        });
        
        // Example sequences
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const sequence = btn.dataset.sequence;
                this.sequenceInput.value = sequence;
                this.validateSequence();
            });
        });
        
        // Fold button
        this.foldButton.addEventListener('click', () => this.foldProtein());
        
        // Download buttons
        this.downloadPDB.addEventListener('click', () => this.downloadPDB());
        this.downloadJSON.addEventListener('click', () => this.downloadJSON());
        
        // Error modal
        this.closeError.addEventListener('click', () => this.hideError());
        this.errorOk.addEventListener('click', () => this.hideError());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    if (this.foldButton.disabled === false) {
                        this.foldProtein();
                    }
                }
            }
        });
    }
    
    async initializeWASM() {
        try {
            this.updateStatus('⏳', 'Loading WASM module...', 'loading');
            
            this.folder = new OdinFoldWASM({
                wasmPath: './odinfold.wasm'
            });
            
            const success = await this.folder.initialize();
            
            if (success) {
                this.updateStatus('✅', 'Ready to fold!', 'ready');
                this.updateModelInfo();
                this.validateSequence(); // Enable fold button if sequence is valid
            } else {
                throw new Error('Failed to initialize WASM module');
            }
            
        } catch (error) {
            console.error('WASM initialization failed:', error);
            this.updateStatus('❌', 'Failed to load', 'error');
            this.showError(`Failed to initialize: ${error.message}`);
        }
    }
    
    updateStatus(indicator, text, className = '') {
        this.statusIndicator.textContent = indicator;
        this.statusText.textContent = text;
        this.status.className = `status ${className}`;
    }
    
    validateSequence() {
        const sequence = this.sequenceInput.value.trim().toUpperCase();
        
        // Update length
        this.sequenceLength.textContent = `Length: ${sequence.length}`;
        
        if (!this.folder) {
            this.foldButton.disabled = true;
            return;
        }
        
        if (sequence.length === 0) {
            this.sequenceValid.textContent = '⚠️ Enter sequence';
            this.sequenceValid.className = 'valid-indicator invalid';
            this.foldButton.disabled = true;
            return;
        }
        
        const validation = this.folder.validateSequence(sequence);
        
        if (validation.valid) {
            this.sequenceValid.textContent = '✅ Valid';
            this.sequenceValid.className = 'valid-indicator';
            this.foldButton.disabled = false;
        } else {
            this.sequenceValid.textContent = `❌ ${validation.error}`;
            this.sequenceValid.className = 'valid-indicator invalid';
            this.foldButton.disabled = true;
        }
    }
    
    async foldProtein() {
        const sequence = this.sequenceInput.value.trim().toUpperCase();
        
        if (!sequence || !this.folder) return;
        
        try {
            // Show progress section
            this.progressSection.style.display = 'block';
            this.resultsSection.style.display = 'none';
            this.foldButton.disabled = true;
            
            // Reset progress
            this.updateProgress(0, 'Initializing...');
            this.startTime = Date.now();
            this.peakMemory = 0;
            
            // Start progress monitoring
            const progressInterval = setInterval(() => {
                const elapsed = (Date.now() - this.startTime) / 1000;
                this.elapsedTime.textContent = `Time: ${elapsed.toFixed(1)}s`;
                
                if (this.folder) {
                    const memory = this.folder.getMemoryUsage();
                    this.peakMemory = Math.max(this.peakMemory, memory);
                    this.memoryUsage.textContent = `Memory: ${memory.toFixed(1)} MB`;
                }
            }, 100);
            
            // Fold protein
            const result = await this.folder.foldProtein(sequence, {
                onProgress: (progress) => {
                    this.updateProgress(progress, this.getProgressMessage(progress));
                }
            });
            
            clearInterval(progressInterval);
            
            // Store result and show results section
            this.currentResult = result;
            this.showResults(result);
            
        } catch (error) {
            console.error('Folding failed:', error);
            this.showError(`Folding failed: ${error.message}`);
            this.progressSection.style.display = 'none';
        } finally {
            this.foldButton.disabled = false;
        }
    }
    
    updateProgress(percent, status) {
        this.progressFill.style.width = `${percent}%`;
        this.progressPercent.textContent = `${Math.round(percent)}%`;
        this.progressStatus.textContent = status;
    }
    
    getProgressMessage(progress) {
        if (progress < 20) return 'Preprocessing sequence...';
        if (progress < 40) return 'Running attention layers...';
        if (progress < 60) return 'Predicting structure...';
        if (progress < 80) return 'Refining coordinates...';
        if (progress < 95) return 'Calculating confidence...';
        return 'Finalizing results...';
    }
    
    showResults(result) {
        // Hide progress, show results
        this.progressSection.style.display = 'none';
        this.resultsSection.style.display = 'block';
        this.resultsSection.classList.add('slide-in');
        
        // Update result summary
        this.resultLength.textContent = `${result.sequenceLength} residues`;
        this.resultConfidence.textContent = `${(result.meanConfidence * 100).toFixed(1)}%`;
        this.resultTime.textContent = `${result.inferenceTimeMs.toFixed(0)} ms`;
        this.resultMemory.textContent = `${result.memoryUsageMb.toFixed(1)} MB`;
        
        // Draw confidence chart
        this.drawConfidenceChart(result.confidence);
        
        // Update performance info
        this.updatePerformanceInfo(result);
        
        console.log('Folding completed:', result);
    }
    
    drawConfidenceChart(confidence) {
        const canvas = this.confidenceChart;
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!confidence || confidence.length === 0) return;
        
        const width = canvas.width;
        const height = canvas.height;
        const padding = 40;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;
        
        // Draw axes
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        
        // Y-axis
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.stroke();
        
        // X-axis
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();
        
        // Draw confidence line
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let i = 0; i < confidence.length; i++) {
            const x = padding + (i / (confidence.length - 1)) * chartWidth;
            const y = height - padding - (confidence[i] * chartHeight);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // Draw labels
        ctx.fillStyle = '#64748b';
        ctx.font = '12px Inter';
        ctx.textAlign = 'center';
        
        // X-axis labels
        ctx.fillText('1', padding, height - padding + 15);
        ctx.fillText(confidence.length.toString(), width - padding, height - padding + 15);
        ctx.fillText('Residue Position', width / 2, height - 5);
        
        // Y-axis labels
        ctx.textAlign = 'right';
        ctx.fillText('0.0', padding - 5, height - padding);
        ctx.fillText('1.0', padding - 5, padding + 5);
        
        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('Confidence', 0, 0);
        ctx.restore();
    }
    
    updatePerformanceInfo(result) {
        const throughput = result.sequenceLength / (result.inferenceTimeMs / 1000);
        
        document.getElementById('lastFoldTime').textContent = `${result.inferenceTimeMs.toFixed(0)} ms`;
        document.getElementById('peakMemory').textContent = `${this.peakMemory.toFixed(1)} MB`;
        document.getElementById('throughput').textContent = `${throughput.toFixed(1)}`;
    }
    
    downloadPDB() {
        if (!this.currentResult) return;
        
        const pdb = this.currentResult.toPDB();
        const blob = new Blob([pdb], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `protein_${Date.now()}.pdb`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    downloadJSON() {
        if (!this.currentResult) return;
        
        const data = {
            sequence: this.currentResult.sequence,
            coordinates: this.currentResult.coordinates,
            confidence: this.currentResult.confidence,
            stats: this.currentResult.stats,
            metadata: {
                timestamp: new Date().toISOString(),
                inference_time_ms: this.currentResult.inferenceTimeMs,
                memory_usage_mb: this.currentResult.memoryUsageMb
            }
        };
        
        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `protein_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    updateSystemInfo() {
        // Browser info
        const browserInfo = `${navigator.userAgent.split(' ').slice(-2).join(' ')}`;
        document.getElementById('browserInfo').textContent = browserInfo;
        
        // WebAssembly support
        const wasmSupport = typeof WebAssembly !== 'undefined' ? 'Supported' : 'Not supported';
        document.getElementById('wasmSupport').textContent = wasmSupport;
        
        // Memory limit (estimate)
        const memoryLimit = navigator.deviceMemory ? `${navigator.deviceMemory} GB` : 'Unknown';
        document.getElementById('memoryLimit').textContent = memoryLimit;
    }
    
    updateModelInfo() {
        if (!this.folder) return;
        
        const info = this.folder.getModelInfo();
        
        document.getElementById('modelVersion').textContent = info.version || '1.0.0';
        document.getElementById('modelMaxLength').textContent = `${info.max_sequence_length || 200} residues`;
        document.getElementById('modelSize').textContent = `${info.model_size_mb || 45} MB`;
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorModal.style.display = 'flex';
    }
    
    hideError() {
        this.errorModal.style.display = 'none';
    }
}

// Initialize demo when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.demo = new OdinFoldDemo();
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.demo && window.demo.folder) {
        window.demo.folder.destroy();
    }
});
