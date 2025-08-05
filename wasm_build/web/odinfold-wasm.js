/**
 * @file odinfold-wasm.js
 * @brief JavaScript wrapper for OdinFold++ WASM module
 * 
 * Provides a clean JavaScript API for protein folding in web browsers.
 * Handles WASM module loading, memory management, and progress tracking.
 */

class OdinFoldWASM {
    constructor(options = {}) {
        this.options = {
            wasmPath: options.wasmPath || './odinfold.wasm',
            maxSequenceLength: options.maxSequenceLength || 200,
            memoryLimit: options.memoryLimit || 512 * 1024 * 1024, // 512MB
            progressCallback: options.progressCallback || null,
            ...options
        };
        
        this.module = null;
        this.folder = null;
        this.initialized = false;
        this.loading = false;
    }
    
    /**
     * Initialize the WASM module and folding engine
     * @returns {Promise<boolean>} Success status
     */
    async initialize() {
        if (this.initialized) return true;
        if (this.loading) {
            // Wait for existing initialization
            while (this.loading) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            return this.initialized;
        }
        
        this.loading = true;
        
        try {
            console.log('Loading OdinFold++ WASM module...');
            
            // Check WebAssembly support
            if (typeof WebAssembly === 'undefined') {
                throw new Error('WebAssembly not supported in this browser');
            }
            
            // Load WASM module
            this.module = await this.loadWASMModule();
            
            // Create folder instance
            this.folder = new this.module.OdinFoldWASM();
            
            // Initialize the folding engine
            const success = this.folder.initialize();
            if (!success) {
                throw new Error('Failed to initialize folding engine');
            }
            
            this.initialized = true;
            console.log('OdinFold++ WASM initialized successfully');
            
            return true;
            
        } catch (error) {
            console.error('Failed to initialize OdinFold++ WASM:', error);
            return false;
        } finally {
            this.loading = false;
        }
    }
    
    /**
     * Load the WASM module
     * @returns {Promise<Object>} WASM module
     */
    async loadWASMModule() {
        return new Promise((resolve, reject) => {
            // Check if Module is already available (for mock)
            if (typeof window !== 'undefined' && window.Module && window.Module.OdinFoldWASM) {
                console.log('Using existing Module');
                resolve(window.Module);
                return;
            }

            const script = document.createElement('script');
            script.src = this.options.wasmPath.replace('.wasm', '.js');

            script.onload = () => {
                // Wait for Module to be available
                const checkModule = () => {
                    if (window.Module && window.Module.OdinFoldWASM) {
                        resolve(window.Module);
                    } else {
                        setTimeout(checkModule, 50);
                    }
                };

                checkModule();
            };

            script.onerror = () => {
                reject(new Error('Failed to load WASM JavaScript file'));
            };

            document.head.appendChild(script);
        });
    }
    
    /**
     * Validate protein sequence
     * @param {string} sequence - Protein sequence
     * @returns {Object} Validation result
     */
    validateSequence(sequence) {
        if (!this.initialized) {
            return { valid: false, error: 'Engine not initialized' };
        }
        
        if (typeof sequence !== 'string') {
            return { valid: false, error: 'Sequence must be a string' };
        }
        
        if (sequence.length === 0) {
            return { valid: false, error: 'Sequence cannot be empty' };
        }
        
        if (sequence.length > this.options.maxSequenceLength) {
            return { 
                valid: false, 
                error: `Sequence too long (${sequence.length} > ${this.options.maxSequenceLength})` 
            };
        }
        
        // Check for invalid characters
        const validAA = /^[ACDEFGHIKLMNPQRSTVWYX]+$/i;
        if (!validAA.test(sequence)) {
            return { valid: false, error: 'Sequence contains invalid amino acids' };
        }
        
        // Use WASM validation
        const isValid = this.folder.validateSequence(sequence.toUpperCase());
        
        return {
            valid: isValid,
            length: sequence.length,
            error: isValid ? null : 'Invalid sequence format'
        };
    }
    
    /**
     * Fold protein sequence
     * @param {string} sequence - Protein sequence
     * @param {Object} options - Folding options
     * @returns {Promise<Object>} Folding result
     */
    async foldProtein(sequence, options = {}) {
        if (!this.initialized) {
            throw new Error('Engine not initialized. Call initialize() first.');
        }
        
        // Validate sequence
        const validation = this.validateSequence(sequence);
        if (!validation.valid) {
            throw new Error(`Invalid sequence: ${validation.error}`);
        }
        
        const progressCallback = options.onProgress || this.options.progressCallback;
        
        try {
            console.log(`Folding protein sequence (${sequence.length} residues)...`);
            
            // Call WASM folding function with direct callback
            const result = await this.folder.foldProtein(
                sequence.toUpperCase(),
                progressCallback
            );
            
            // Process result
            if (!result.success) {
                throw new Error(result.error || 'Folding failed');
            }
            
            // Convert coordinates to more usable format
            const processedResult = this.processResult(result, sequence);
            
            console.log(`Folding completed in ${result.inference_time_ms}ms`);
            
            return processedResult;
            
        } catch (error) {
            console.error('Folding failed:', error);
            throw error;
        }
    }
    
    /**
     * Process raw WASM result into JavaScript-friendly format
     * @param {Object} rawResult - Raw result from WASM
     * @param {string} sequence - Original sequence
     * @returns {Object} Processed result
     */
    processResult(rawResult, sequence) {
        return {
            success: true,
            sequence: sequence,
            sequenceLength: rawResult.sequence_length,
            
            // Coordinates as array of [x, y, z] arrays
            coordinates: rawResult.coordinates,
            
            // Confidence scores
            confidence: rawResult.confidence,
            meanConfidence: rawResult.confidence.reduce((a, b) => a + b, 0) / rawResult.confidence.length,
            
            // Performance metrics
            inferenceTimeMs: rawResult.inference_time_ms,
            memoryUsageMb: rawResult.memory_usage_mb,
            
            // Utility methods
            toPDB: () => this.generatePDB(rawResult.coordinates, sequence, rawResult.confidence),
            
            // Statistics
            stats: {
                highConfidenceResidues: rawResult.confidence.filter(c => c > 0.8).length,
                lowConfidenceResidues: rawResult.confidence.filter(c => c < 0.5).length,
                averageConfidence: rawResult.confidence.reduce((a, b) => a + b, 0) / rawResult.confidence.length
            }
        };
    }
    
    /**
     * Generate PDB format string
     * @param {Array} coordinates - Protein coordinates
     * @param {string} sequence - Protein sequence
     * @param {Array} confidence - Confidence scores
     * @returns {string} PDB format string
     */
    generatePDB(coordinates, sequence, confidence) {
        if (!this.initialized) {
            throw new Error('Engine not initialized');
        }
        
        return this.folder.generatePDB(coordinates, sequence, confidence);
    }
    
    /**
     * Get model information
     * @returns {Object} Model information
     */
    getModelInfo() {
        if (!this.initialized) {
            return { error: 'Engine not initialized' };
        }
        
        return this.folder.getModelInfo();
    }
    
    /**
     * Get current memory usage
     * @returns {number} Memory usage in MB
     */
    getMemoryUsage() {
        if (!this.initialized) return 0;
        return this.folder.getMemoryUsageMB();
    }
    
    /**
     * Check if engine is ready
     * @returns {boolean} Ready status
     */
    isReady() {
        return this.initialized && this.folder && this.folder.isInitialized();
    }
    
    /**
     * Get maximum supported sequence length
     * @returns {number} Maximum sequence length
     */
    getMaxSequenceLength() {
        if (!this.initialized) return this.options.maxSequenceLength;
        return this.folder.getMaxSequenceLength();
    }
    
    /**
     * Cleanup resources
     */
    destroy() {
        if (this.folder) {
            this.folder.delete();
            this.folder = null;
        }
        
        this.initialized = false;
        console.log('OdinFold++ WASM resources cleaned up');
    }
}

// Web Worker support
if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {
    // Running in Web Worker
    let folder = null;
    
    self.onmessage = async function(e) {
        const { action, data, id } = e.data;
        
        try {
            switch (action) {
                case 'initialize':
                    folder = new OdinFoldWASM(data.options);
                    const success = await folder.initialize();
                    self.postMessage({ id, success });
                    break;
                    
                case 'fold':
                    if (!folder) {
                        throw new Error('Folder not initialized');
                    }
                    
                    const result = await folder.foldProtein(data.sequence, {
                        onProgress: (progress) => {
                            self.postMessage({ 
                                id, 
                                type: 'progress', 
                                progress 
                            });
                        }
                    });
                    
                    self.postMessage({ id, type: 'result', result });
                    break;
                    
                case 'validate':
                    if (!folder) {
                        throw new Error('Folder not initialized');
                    }
                    
                    const validation = folder.validateSequence(data.sequence);
                    self.postMessage({ id, validation });
                    break;
                    
                default:
                    throw new Error(`Unknown action: ${action}`);
            }
            
        } catch (error) {
            self.postMessage({ 
                id, 
                error: error.message 
            });
        }
    };
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = OdinFoldWASM;
} else if (typeof define === 'function' && define.amd) {
    define([], function() { return OdinFoldWASM; });
} else {
    window.OdinFoldWASM = OdinFoldWASM;
}

// Auto-initialize if requested
if (typeof window !== 'undefined' && window.ODINFOLD_AUTO_INIT) {
    window.addEventListener('DOMContentLoaded', async () => {
        window.odinFold = new OdinFoldWASM();
        await window.odinFold.initialize();
        console.log('OdinFold++ WASM auto-initialized');
    });
}
