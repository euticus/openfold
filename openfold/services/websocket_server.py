"""
WebSocket server for real-time protein structure mutation.

This module provides a FastAPI-based WebSocket server that enables
persistent sessions for interactive protein structure editing and mutation.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import traceback

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. WebSocket server functionality disabled.")

import torch
import numpy as np
from dataclasses import dataclass, asdict
from openfold.np import protein
from openfold.model.delta_predictor import DeltaPredictor, MutationInput, create_delta_predictor


@dataclass
class MutationRequest:
    """Request for applying a mutation."""
    position: int  # 0-indexed residue position
    original_aa: str  # Original amino acid (single letter)
    target_aa: str   # Target amino acid (single letter)
    session_id: str
    request_id: str = None


@dataclass
class MutationResponse:
    """Response after applying a mutation."""
    success: bool
    request_id: str
    session_id: str
    mutation: str  # e.g., "A42V"
    updated_structure: Optional[str] = None  # PDB string
    position_deltas: Optional[List[List[float]]] = None  # [[x,y,z], ...]
    confidence_scores: Optional[List[float]] = None
    affected_residues: Optional[List[int]] = None
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None


@dataclass
class SessionInfo:
    """Information about a WebSocket session."""
    session_id: str
    websocket: WebSocket
    original_structure: protein.Protein
    current_structure: protein.Protein
    mutation_history: List[MutationRequest]
    created_at: datetime
    last_activity: datetime


class StructureSession:
    """Manages a single protein structure editing session."""
    
    def __init__(self, 
                 session_id: str,
                 original_structure: protein.Protein,
                 delta_predictor: DeltaPredictor):
        """
        Args:
            session_id: Unique session identifier
            original_structure: Initial protein structure
            delta_predictor: Model for predicting mutation effects
        """
        self.session_id = session_id
        self.original_structure = original_structure
        self.current_structure = original_structure
        self.delta_predictor = delta_predictor
        self.mutation_history = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def apply_mutation(self, mutation_request: MutationRequest) -> MutationResponse:
        """
        Apply a mutation to the current structure.
        
        Args:
            mutation_request: Mutation to apply
            
        Returns:
            Response with updated structure and metadata
        """
        start_time = datetime.now()
        
        try:
            # Validate mutation request
            if mutation_request.position >= len(self.current_structure.aatype):
                raise ValueError(f"Position {mutation_request.position} out of range")
            
            # Check if mutation is valid
            current_aa = protein.residue_constants.restypes[
                self.current_structure.aatype[mutation_request.position]
            ]
            
            if current_aa != mutation_request.original_aa:
                logging.warning(
                    f"Original AA mismatch: expected {mutation_request.original_aa}, "
                    f"found {current_aa} at position {mutation_request.position}"
                )
            
            # Create mutation input for delta predictor
            mutation_input = MutationInput(
                protein_structure=self.current_structure,
                mutation_position=mutation_request.position,
                original_aa=mutation_request.original_aa,
                target_aa=mutation_request.target_aa,
                local_radius=10.0
            )
            
            # Predict structural changes
            with torch.no_grad():
                prediction = self.delta_predictor(mutation_input)
            
            # Apply predicted changes to structure
            updated_structure = self._apply_delta_prediction(
                self.current_structure,
                prediction,
                mutation_request
            )
            
            # Update session state
            self.current_structure = updated_structure
            self.mutation_history.append(mutation_request)
            self.last_activity = datetime.now()
            
            # Create response
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            mutation_str = f"{mutation_request.original_aa}{mutation_request.position+1}{mutation_request.target_aa}"
            
            response = MutationResponse(
                success=True,
                request_id=mutation_request.request_id or str(uuid.uuid4()),
                session_id=self.session_id,
                mutation=mutation_str,
                updated_structure=protein.to_pdb(updated_structure),
                position_deltas=prediction.position_deltas.tolist(),
                confidence_scores=prediction.confidence_scores.tolist(),
                affected_residues=prediction.affected_residues,
                processing_time_ms=processing_time
            )
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return MutationResponse(
                success=False,
                request_id=mutation_request.request_id or str(uuid.uuid4()),
                session_id=self.session_id,
                mutation=f"{mutation_request.original_aa}{mutation_request.position+1}{mutation_request.target_aa}",
                processing_time_ms=processing_time,
                error_message=str(e)
            )
    
    def _apply_delta_prediction(self, 
                               structure: protein.Protein,
                               prediction,
                               mutation_request: MutationRequest) -> protein.Protein:
        """Apply predicted structural changes to create updated structure."""
        # Create a copy of the structure
        new_positions = structure.atom_positions.copy()
        new_aatype = structure.aatype.copy()
        
        # Update amino acid type
        target_aa_idx = protein.residue_constants.restype_order.get(
            mutation_request.target_aa, 20
        )
        new_aatype[mutation_request.position] = target_aa_idx
        
        # Apply position deltas to affected residues
        delta_tensor = prediction.position_deltas
        affected_residues = prediction.affected_residues
        
        # Map deltas back to full structure
        for i, res_idx in enumerate(affected_residues):
            if res_idx < len(new_positions):
                # Apply deltas to atoms (simplified - assumes same atom ordering)
                atoms_per_res = 37  # Standard atom count
                start_atom = i * atoms_per_res
                end_atom = min(start_atom + atoms_per_res, len(delta_tensor))
                
                if end_atom > start_atom:
                    atom_deltas = delta_tensor[start_atom:end_atom]
                    for j, delta in enumerate(atom_deltas):
                        if j < new_positions.shape[1]:  # Within atom bounds
                            new_positions[res_idx, j] += delta.numpy()
        
        # Create updated protein
        updated_protein = protein.Protein(
            atom_positions=new_positions,
            atom_mask=structure.atom_mask,
            aatype=new_aatype,
            residue_index=structure.residue_index,
            b_factors=structure.b_factors,
            chain_index=getattr(structure, 'chain_index', None),
            remark=f"Mutated: {mutation_request.original_aa}{mutation_request.position+1}{mutation_request.target_aa}"
        )
        
        return updated_protein
    
    def reset_structure(self) -> None:
        """Reset structure to original state."""
        self.current_structure = self.original_structure
        self.mutation_history = []
        self.last_activity = datetime.now()
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information."""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'mutation_count': len(self.mutation_history),
            'current_sequence_length': len(self.current_structure.aatype),
            'mutations_applied': [
                f"{m.original_aa}{m.position+1}{m.target_aa}" 
                for m in self.mutation_history
            ]
        }


class WebSocketMutationServer:
    """WebSocket server for real-time protein mutation."""
    
    def __init__(self, 
                 delta_predictor: Optional[DeltaPredictor] = None,
                 session_timeout_minutes: int = 60):
        """
        Args:
            delta_predictor: Model for predicting mutation effects
            session_timeout_minutes: Session timeout in minutes
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for WebSocket server")
        
        self.delta_predictor = delta_predictor or create_delta_predictor()
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
        # Session management
        self.sessions: Dict[str, StructureSession] = {}
        self.websocket_sessions: Dict[str, WebSocket] = {}
        
        # Create FastAPI app
        self.app = FastAPI(
            title="OpenFold++ Mutation Server",
            description="Real-time protein structure mutation via WebSocket",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()

        # Cleanup task will be started when server runs
        self._cleanup_task = None
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/")
        async def root():
            return {"message": "OpenFold++ Mutation Server", "status": "running"}
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "active_sessions": len(self.sessions),
                "model_loaded": self.delta_predictor is not None
            }
        
        @self.app.get("/sessions")
        async def list_sessions():
            """List all active sessions."""
            return {
                "sessions": [
                    session.get_session_info() 
                    for session in self.sessions.values()
                ]
            }
        
        @self.app.websocket("/ws/mutate/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            await self._handle_websocket_connection(websocket, session_id)
        
        @self.app.get("/demo")
        async def demo_page():
            """Serve a simple demo page."""
            return HTMLResponse(self._get_demo_html())
    
    async def _handle_websocket_connection(self, websocket: WebSocket, session_id: str):
        """Handle WebSocket connection for mutation requests."""
        await websocket.accept()
        
        try:
            # Register WebSocket
            self.websocket_sessions[session_id] = websocket
            
            # Send welcome message
            await websocket.send_json({
                "type": "connection",
                "message": f"Connected to session {session_id}",
                "session_id": session_id
            })
            
            # Handle messages
            while True:
                try:
                    # Receive message
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Process message
                    response = await self._process_websocket_message(message, session_id)
                    
                    # Send response
                    await websocket.send_json(asdict(response))
                    
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON format"
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error processing message: {str(e)}"
                    })
        
        except Exception as e:
            logging.error(f"WebSocket error for session {session_id}: {e}")
        
        finally:
            # Cleanup
            if session_id in self.websocket_sessions:
                del self.websocket_sessions[session_id]
    
    async def _process_websocket_message(self, message: Dict[str, Any], session_id: str) -> MutationResponse:
        """Process incoming WebSocket message."""
        message_type = message.get("type", "unknown")
        
        if message_type == "init_session":
            return await self._init_session(message, session_id)
        elif message_type == "mutate":
            return await self._handle_mutation(message, session_id)
        elif message_type == "reset":
            return await self._reset_session(session_id)
        elif message_type == "get_info":
            return await self._get_session_info(session_id)
        else:
            return MutationResponse(
                success=False,
                request_id=message.get("request_id", "unknown"),
                session_id=session_id,
                mutation="",
                error_message=f"Unknown message type: {message_type}"
            )
    
    async def _init_session(self, message: Dict[str, Any], session_id: str) -> MutationResponse:
        """Initialize a new session with a protein structure."""
        try:
            # Parse PDB string
            pdb_string = message.get("pdb_string", "")
            if not pdb_string:
                raise ValueError("PDB string is required for session initialization")
            
            # Create protein object
            structure = protein.from_pdb_string(pdb_string)
            
            # Create session
            session = StructureSession(
                session_id=session_id,
                original_structure=structure,
                delta_predictor=self.delta_predictor
            )
            
            self.sessions[session_id] = session
            
            return MutationResponse(
                success=True,
                request_id=message.get("request_id", str(uuid.uuid4())),
                session_id=session_id,
                mutation="session_initialized",
                updated_structure=pdb_string
            )
            
        except Exception as e:
            return MutationResponse(
                success=False,
                request_id=message.get("request_id", str(uuid.uuid4())),
                session_id=session_id,
                mutation="session_init_failed",
                error_message=str(e)
            )
    
    async def _handle_mutation(self, message: Dict[str, Any], session_id: str) -> MutationResponse:
        """Handle mutation request."""
        if session_id not in self.sessions:
            return MutationResponse(
                success=False,
                request_id=message.get("request_id", str(uuid.uuid4())),
                session_id=session_id,
                mutation="",
                error_message="Session not found. Please initialize session first."
            )
        
        try:
            # Create mutation request
            mutation_request = MutationRequest(
                position=message["position"],
                original_aa=message["original_aa"],
                target_aa=message["target_aa"],
                session_id=session_id,
                request_id=message.get("request_id", str(uuid.uuid4()))
            )
            
            # Apply mutation
            session = self.sessions[session_id]
            response = session.apply_mutation(mutation_request)
            
            return response
            
        except KeyError as e:
            return MutationResponse(
                success=False,
                request_id=message.get("request_id", str(uuid.uuid4())),
                session_id=session_id,
                mutation="",
                error_message=f"Missing required field: {e}"
            )
        except Exception as e:
            return MutationResponse(
                success=False,
                request_id=message.get("request_id", str(uuid.uuid4())),
                session_id=session_id,
                mutation="",
                error_message=str(e)
            )
    
    async def _reset_session(self, session_id: str) -> MutationResponse:
        """Reset session to original structure."""
        if session_id not in self.sessions:
            return MutationResponse(
                success=False,
                request_id=str(uuid.uuid4()),
                session_id=session_id,
                mutation="",
                error_message="Session not found"
            )
        
        session = self.sessions[session_id]
        session.reset_structure()
        
        return MutationResponse(
            success=True,
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            mutation="session_reset",
            updated_structure=protein.to_pdb(session.current_structure)
        )
    
    async def _get_session_info(self, session_id: str) -> MutationResponse:
        """Get session information."""
        if session_id not in self.sessions:
            return MutationResponse(
                success=False,
                request_id=str(uuid.uuid4()),
                session_id=session_id,
                mutation="",
                error_message="Session not found"
            )
        
        session = self.sessions[session_id]
        info = session.get_session_info()
        
        return MutationResponse(
            success=True,
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            mutation="session_info",
            updated_structure=json.dumps(info)
        )
    
    async def _cleanup_sessions(self):
        """Periodically cleanup expired sessions."""
        while True:
            try:
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if current_time - session.last_activity > self.session_timeout:
                        expired_sessions.append(session_id)
                
                # Remove expired sessions
                for session_id in expired_sessions:
                    del self.sessions[session_id]
                    if session_id in self.websocket_sessions:
                        del self.websocket_sessions[session_id]
                    logging.info(f"Cleaned up expired session: {session_id}")
                
                # Wait before next cleanup
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logging.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    def _get_demo_html(self) -> str:
        """Get demo HTML page."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>OpenFold++ Mutation Demo</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .input-group { margin: 10px 0; }
                label { display: inline-block; width: 120px; }
                input, textarea { width: 300px; padding: 5px; }
                button { padding: 10px 20px; margin: 5px; }
                .output { background: #f5f5f5; padding: 20px; margin: 20px 0; }
                .error { color: red; }
                .success { color: green; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>OpenFold++ Mutation Demo</h1>
                <p>Real-time protein structure mutation via WebSocket</p>
                
                <div class="input-group">
                    <label>Session ID:</label>
                    <input type="text" id="sessionId" value="demo-session" />
                    <button onclick="connect()">Connect</button>
                    <button onclick="disconnect()">Disconnect</button>
                </div>
                
                <div class="input-group">
                    <label>PDB String:</label>
                    <textarea id="pdbString" rows="5" placeholder="Paste PDB content here..."></textarea>
                    <button onclick="initSession()">Initialize Session</button>
                </div>
                
                <div class="input-group">
                    <label>Position:</label>
                    <input type="number" id="position" value="0" />
                </div>
                
                <div class="input-group">
                    <label>Original AA:</label>
                    <input type="text" id="originalAA" value="A" maxlength="1" />
                </div>
                
                <div class="input-group">
                    <label>Target AA:</label>
                    <input type="text" id="targetAA" value="V" maxlength="1" />
                </div>
                
                <button onclick="applyMutation()">Apply Mutation</button>
                <button onclick="resetSession()">Reset Session</button>
                <button onclick="getSessionInfo()">Get Session Info</button>
                
                <div class="output">
                    <h3>Output:</h3>
                    <div id="output"></div>
                </div>
            </div>
            
            <script>
                let ws = null;
                
                function connect() {
                    const sessionId = document.getElementById('sessionId').value;
                    ws = new WebSocket(`ws://localhost:8000/ws/mutate/${sessionId}`);
                    
                    ws.onopen = function(event) {
                        log('Connected to WebSocket', 'success');
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        log('Received: ' + JSON.stringify(data, null, 2), data.success ? 'success' : 'error');
                    };
                    
                    ws.onclose = function(event) {
                        log('WebSocket connection closed', 'error');
                    };
                    
                    ws.onerror = function(error) {
                        log('WebSocket error: ' + error, 'error');
                    };
                }
                
                function disconnect() {
                    if (ws) {
                        ws.close();
                        ws = null;
                        log('Disconnected', 'success');
                    }
                }
                
                function initSession() {
                    if (!ws) {
                        log('Please connect first', 'error');
                        return;
                    }
                    
                    const message = {
                        type: 'init_session',
                        pdb_string: document.getElementById('pdbString').value,
                        request_id: generateId()
                    };
                    
                    ws.send(JSON.stringify(message));
                }
                
                function applyMutation() {
                    if (!ws) {
                        log('Please connect first', 'error');
                        return;
                    }
                    
                    const message = {
                        type: 'mutate',
                        position: parseInt(document.getElementById('position').value),
                        original_aa: document.getElementById('originalAA').value,
                        target_aa: document.getElementById('targetAA').value,
                        request_id: generateId()
                    };
                    
                    ws.send(JSON.stringify(message));
                }
                
                function resetSession() {
                    if (!ws) {
                        log('Please connect first', 'error');
                        return;
                    }
                    
                    const message = {
                        type: 'reset',
                        request_id: generateId()
                    };
                    
                    ws.send(JSON.stringify(message));
                }
                
                function getSessionInfo() {
                    if (!ws) {
                        log('Please connect first', 'error');
                        return;
                    }
                    
                    const message = {
                        type: 'get_info',
                        request_id: generateId()
                    };
                    
                    ws.send(JSON.stringify(message));
                }
                
                function log(message, type = '') {
                    const output = document.getElementById('output');
                    const div = document.createElement('div');
                    div.className = type;
                    div.innerHTML = '<strong>' + new Date().toLocaleTimeString() + ':</strong> ' + message;
                    output.appendChild(div);
                    output.scrollTop = output.scrollHeight;
                }
                
                function generateId() {
                    return Math.random().toString(36).substr(2, 9);
                }
            </script>
        </body>
        </html>
        """
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the WebSocket server."""
        # Start cleanup task when server starts
        @self.app.on_event("startup")
        async def startup_event():
            self._cleanup_task = asyncio.create_task(self._cleanup_sessions())

        @self.app.on_event("shutdown")
        async def shutdown_event():
            if self._cleanup_task:
                self._cleanup_task.cancel()

        uvicorn.run(self.app, host=host, port=port, **kwargs)


def create_mutation_server(delta_predictor: Optional[DeltaPredictor] = None) -> WebSocketMutationServer:
    """
    Factory function to create WebSocket mutation server.
    
    Args:
        delta_predictor: Optional delta prediction model
        
    Returns:
        WebSocketMutationServer instance
    """
    return WebSocketMutationServer(delta_predictor=delta_predictor)
