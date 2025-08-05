/**
 * Mock Emscripten-generated JavaScript for OdinFold++ WASM
 * This simulates the actual WASM module for demo purposes
 */

// Mock Module object
var Module = typeof Module !== 'undefined' ? Module : {};

// Mock WASM functions
Module.onRuntimeInitialized = function() {
    console.log('🧬 OdinFold++ WASM Runtime Initialized (Mock)');
    
    // Simulate the C++ class binding
    Module.OdinFoldWASM = function() {
        this.initialized = false;
        this.maxSeqLen = 200;
        
        // Mock amino acid mapping
        this.aaToIdx = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
            'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20
        };
    };
    
    Module.OdinFoldWASM.prototype.initialize = function() {
        console.log('Initializing mock WASM folding engine...');
        this.initialized = true;
        return true;
    };
    
    Module.OdinFoldWASM.prototype.isInitialized = function() {
        return this.initialized;
    };
    
    Module.OdinFoldWASM.prototype.getMaxSequenceLength = function() {
        return this.maxSeqLen;
    };
    
    Module.OdinFoldWASM.prototype.validateSequence = function(sequence) {
        if (!sequence || sequence.length === 0) return false;
        if (sequence.length > this.maxSeqLen) return false;
        
        // Check for valid amino acids
        for (let i = 0; i < sequence.length; i++) {
            if (!(sequence[i].toUpperCase() in this.aaToIdx)) {
                return false;
            }
        }
        return true;
    };
    
    Module.OdinFoldWASM.prototype.foldProtein = function(sequence, progressCallback) {
        console.log(`Mock folding protein: ${sequence.substring(0, 20)}... (${sequence.length} residues)`);

        const self = this; // Capture 'this' context

        return new Promise((resolve, reject) => {
            try {
                let progress = 0;
                const interval = setInterval(() => {
                    progress += Math.random() * 15 + 5; // Random progress increments

                    if (progress >= 100) {
                        progress = 100;
                        clearInterval(interval);

                        try {
                            // Generate mock results
                            const coordinates = self.generateMockCoordinates(sequence.length);
                            const confidence = self.generateMockConfidence(sequence.length);

                            const result = {
                                success: true,
                                sequence: sequence,
                                sequence_length: sequence.length,
                                coordinates: coordinates,
                                confidence: confidence,
                                inference_time_ms: 8000 + Math.random() * 4000, // 8-12 seconds
                                memory_usage_mb: 200 + sequence.length * 1.5
                            };

                            if (progressCallback && typeof progressCallback === 'function') {
                                progressCallback(100);
                            }

                            setTimeout(() => resolve(result), 500);
                        } catch (error) {
                            console.error('Error generating mock results:', error);
                            reject(error);
                        }
                    } else {
                        if (progressCallback && typeof progressCallback === 'function') {
                            progressCallback(Math.floor(progress));
                        }
                    }
                }, 200); // Update every 200ms
            } catch (error) {
                console.error('Error in foldProtein:', error);
                reject(error);
            }
        });
    };
    
    Module.OdinFoldWASM.prototype.generateMockCoordinates = function(length) {
        const coordinates = [];
        let x = 0, y = 0, z = 0;
        
        for (let i = 0; i < length; i++) {
            // Generate realistic-looking protein backbone
            const angle = i * 0.1;
            x += 3.8 * Math.cos(angle) + (Math.random() - 0.5) * 0.5;
            y += 3.8 * Math.sin(angle) + (Math.random() - 0.5) * 0.5;
            z += (Math.random() - 0.5) * 0.8;
            
            coordinates.push([x, y, z]);
        }
        
        return coordinates;
    };
    
    Module.OdinFoldWASM.prototype.generateMockConfidence = function(length) {
        const confidence = [];
        
        for (let i = 0; i < length; i++) {
            // Higher confidence in middle, lower at ends
            const posFactor = 1.0 - Math.abs(i - length / 2.0) / (length / 2.0);
            const baseConfidence = 0.7 + 0.2 * posFactor;
            const noise = (Math.random() - 0.5) * 0.1;
            const conf = Math.max(0.0, Math.min(1.0, baseConfidence + noise));
            
            confidence.push(conf);
        }
        
        return confidence;
    };
    
    Module.OdinFoldWASM.prototype.generatePDB = function(coordinates, sequence, confidence) {
        let pdb = '';
        
        // PDB header
        pdb += 'HEADER    PROTEIN STRUCTURE PREDICTION             01-JAN-24   WASM\n';
        pdb += 'TITLE     STRUCTURE PREDICTED BY ODINFOLD++ WASM (MOCK)\n';
        pdb += 'REMARK   1 PREDICTION METHOD: ODINFOLD++ WEBASSEMBLY MOCK\n';
        
        const avgConfidence = confidence.reduce((a, b) => a + b, 0) / confidence.length;
        pdb += `REMARK   1 AVERAGE CONFIDENCE: ${avgConfidence.toFixed(3)}\n`;
        
        // Atom records
        for (let i = 0; i < sequence.length && i < coordinates.length; i++) {
            const aa = sequence[i];
            const [x, y, z] = coordinates[i];
            const bFactor = confidence[i] * 100.0;
            
            const atomLine = `ATOM  ${(i + 1).toString().padStart(5)}  CA  ${aa} A${(i + 1).toString().padStart(4)}    ${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}${(1.00).toFixed(2).padStart(6)}${bFactor.toFixed(2).padStart(6)}           C\n`;
            pdb += atomLine;
        }
        
        pdb += 'END\n';
        return pdb;
    };
    
    Module.OdinFoldWASM.prototype.getMemoryUsageMB = function() {
        return 128 + Math.random() * 50; // Mock memory usage
    };
    
    Module.OdinFoldWASM.prototype.getModelInfo = function() {
        return {
            name: 'OdinFold++ WASM (Mock)',
            version: '1.0.0',
            max_sequence_length: this.maxSeqLen,
            model_size_mb: 45.2,
            supported_features: ['single_chain_folding', 'confidence_scoring', 'pdb_output']
        };
    };
    
    // Mock delete method
    Module.OdinFoldWASM.prototype.delete = function() {
        console.log('Mock WASM object deleted');
    };
    
    // Mock function management
    Module.addFunction = function(func, signature) {
        return func; // Just return the function for mock
    };
    
    Module.removeFunction = function(func) {
        // No-op for mock
    };
    
    // Trigger initialization callback
    if (typeof window !== 'undefined' && window.Module && window.Module.onRuntimeInitialized) {
        window.Module.onRuntimeInitialized();
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    window.Module = Module;

    // Initialize immediately for mock
    if (Module.onRuntimeInitialized) {
        Module.onRuntimeInitialized();
    }
}

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Module;
}
