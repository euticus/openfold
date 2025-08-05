#!/usr/bin/env python3
"""
Simple HTTP server for OdinFold++ WASM demo
Serves the web demo with proper MIME types for WASM files
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

class WASMHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler with proper MIME types for WASM files."""
    
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        super().end_headers()
    
    def guess_type(self, path):
        """Override to add WASM MIME type."""
        if path.endswith('.wasm'):
            return 'application/wasm'
        elif path.endswith('.js'):
            return 'application/javascript'

        return super().guess_type(path)

def main():
    # Change to web directory
    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        os.chdir(web_dir)
        print(f"Serving from: {web_dir}")
    else:
        print(f"Web directory not found: {web_dir}")
        print("Please run from the wasm_build directory")
        return 1
    
    # Start server
    PORT = 8000
    
    try:
        with socketserver.TCPServer(("", PORT), WASMHandler) as httpd:
            print(f"🧬 OdinFold++ WASM Demo Server")
            print(f"📡 Server running at: http://localhost:{PORT}")
            print(f"🌐 Open in browser: http://localhost:{PORT}")
            print(f"⏹️  Press Ctrl+C to stop")
            print()
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        return 0
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use")
            print(f"Try a different port or stop the existing server")
            return 1
        else:
            print(f"❌ Server error: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(main())
