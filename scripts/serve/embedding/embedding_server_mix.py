#!/usr/bin/env python3
"""
Embedding Server for RT-Cache System (Mix View: Third-Person + Wrist)

This FastAPI server generates vision-language embeddings using OpenVLA model.
It accepts two images (third-person view and wrist/elbow view) and concatenates
their DINOv2 + SigLIP features to create a 4352-dimensional embedding.

Embedding structure:
- Third-person DINOv2: 1024 dims
- Third-person SigLIP: 1152 dims  
- Wrist DINOv2: 1024 dims
- Wrist SigLIP: 1152 dims
- Total: 4352 dims

Author: RT-Cache Team
Date: 2024
"""

import os
import sys
import logging
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from io import BytesIO
import time

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)
from pydantic import BaseModel
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "config"))

# Load centralized configuration
from rt_cache_config import get_config


def setup_logging(level="INFO"):
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# ============================================================================
# Configuration
# ============================================================================

class ServerConfig:
    """Server configuration using centralized config"""
    
    def __init__(self):
        config = get_config()
        
        # Server settings - use different port for mix server
        self.host = config.server.embedding_host
        self.port = 9021  # Different port from original (9020)
        self.workers = config.server.embedding_workers
        
        # Model settings
        self.device = config.model.device
        self.dtype = config.model.model_dtype
        self.use_flash_attention = config.model.use_flash_attention
        
        # Processing settings
        self.max_batch_size = config.model.model_batch_size
        self.image_size = (224, 224)
        
        # Logging
        self.log_level = config.paths.log_level
        
        # Mix embedding dimension: (DINOv2 + SigLIP) * 2 views = 4352
        self.mix_embedding_dim = 4352


# ============================================================================
# Response Models
# ============================================================================

class MixEmbeddingResponse(BaseModel):
    """Response model for mix embedding endpoint"""
    
    # Combined features from both views
    mix_features: Optional[str] = None  # Base64 encoded tensor (4352 dims)
    
    # Individual view features (optional, for debugging)
    third_person_features: Optional[str] = None  # Base64 encoded (2176 dims)
    wrist_features: Optional[str] = None  # Base64 encoded (2176 dims)
    
    # Metadata
    processing_time: float
    model_versions: Dict[str, str]


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    
    status: str
    models_loaded: Dict[str, bool]
    device: str
    memory_usage: Dict[str, float]


# ============================================================================
# Mix Embedding Server
# ============================================================================

class MixEmbeddingServer:
    """
    FastAPI server for generating mix vision embeddings.
    
    This server provides:
    - Combined DINOv2 + SigLIP embeddings from two camera views
    - Third-person view (main camera)
    - Wrist/elbow view (arm camera)
    - Total embedding dimension: 4352
    """
    
    def __init__(self, config: ServerConfig):
        """
        Initialize the embedding server.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="RT-Cache Mix Embedding Server",
            description="Mix vision embedding generation for robot control (third-person + wrist view)",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self._setup_cors()
        
        # Load models
        self._load_models()
        
        # Setup routes
        self._setup_routes()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_image_pairs": 0,
            "avg_processing_time": 0
        }
        
    def _setup_cors(self):
        """Configure CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        
    def _load_models(self):
        """Load OpenVLA model for feature extraction"""
        self.logger.info("Loading models...")
        
        try:
            # Load OpenVLA
            self.logger.info("Loading OpenVLA model...")
            self.openvla_processor = AutoProcessor.from_pretrained(
                "openvla/openvla-7b",
                trust_remote_code=True
            )
            
            # Determine dtype
            if self.config.dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            elif self.config.dtype == "float16":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            # Use device_map to load directly to GPU (required for Flash Attention 2.0)
            self.logger.info(f"Loading model to device: {self.config.device}")
            self.openvla_model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b",
                attn_implementation="flash_attention_2" if self.config.use_flash_attention else "eager",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=self.config.device  # Load directly to specified device
            )
            
            self.logger.info("OpenVLA model loaded successfully")
            
            # Model info (all values must be strings for Pydantic validation)
            self.model_info = {
                "openvla": "openvla/openvla-7b",
                "device": self.config.device,
                "dtype": str(torch_dtype),
                "mix_embedding_dim": str(self.config.mix_embedding_dim)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
            
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint"""
            return {
                "service": "RT-Cache Mix Embedding Server",
                "version": "1.0.0",
                "status": "running",
                "embedding_dim": str(self.config.mix_embedding_dim)
            }
            
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return self._get_health_status()
            
        @self.app.post("/predict", response_model=MixEmbeddingResponse)
        async def generate_mix_embeddings(
            third_person_image: UploadFile = File(..., description="Third-person view image"),
            wrist_image: UploadFile = File(..., description="Wrist/elbow view image"),
            instruction: str = Form("", description="Optional text instruction"),
            return_individual: bool = Form(False, description="Return individual view features")
        ):
            """
            Generate mix embeddings for two-view input.
            
            Args:
                third_person_image: Third-person camera view image
                wrist_image: Wrist/elbow camera view image
                instruction: Optional text instruction
                return_individual: Whether to return individual view features
                
            Returns:
                MixEmbeddingResponse with base64-encoded embeddings
            """
            return await self._process_mix_embedding_request(
                third_person_image, wrist_image, instruction, return_individual
            )
            
        @self.app.get("/stats")
        async def get_statistics():
            """Get server statistics"""
            return self.stats
            
    async def _process_mix_embedding_request(
        self,
        third_person_image: UploadFile,
        wrist_image: UploadFile,
        instruction: str,
        return_individual: bool
    ) -> MixEmbeddingResponse:
        """
        Process mix embedding generation request.
        
        Args:
            third_person_image: Third-person view image file
            wrist_image: Wrist view image file
            instruction: Text instruction
            return_individual: Whether to return individual features
            
        Returns:
            MixEmbeddingResponse with embeddings
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        self.stats["total_image_pairs"] += 1
        
        try:
            # Load images
            third_person_pil = await self._load_image(third_person_image)
            wrist_pil = await self._load_image(wrist_image)
            
            # Prepare prompt for OpenVLA
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            
            # Generate embeddings for both views
            third_person_embedding = self._generate_view_embedding(third_person_pil, prompt)
            wrist_embedding = self._generate_view_embedding(wrist_pil, prompt)
            
            if third_person_embedding is None or wrist_embedding is None:
                raise HTTPException(status_code=500, detail="Failed to generate embeddings")
            
            # Concatenate features: [third_person_dino, third_person_siglip, wrist_dino, wrist_siglip]
            mix_features = torch.cat([third_person_embedding, wrist_embedding], dim=-1)
            
            # Prepare response
            result = {
                "mix_features": self._encode_tensor_to_base64(mix_features)
            }
            
            if return_individual:
                result["third_person_features"] = self._encode_tensor_to_base64(third_person_embedding)
                result["wrist_features"] = self._encode_tensor_to_base64(wrist_embedding)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update average processing time
            n = self.stats["total_requests"]
            self.stats["avg_processing_time"] = (
                (self.stats["avg_processing_time"] * (n - 1) + processing_time) / n
            )
            
            return MixEmbeddingResponse(
                **result,
                processing_time=processing_time,
                model_versions=self.model_info
            )
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _generate_view_embedding(self, image: Image.Image, prompt: str) -> Optional[torch.Tensor]:
        """
        Generate DINOv2 + SigLIP embedding for a single view.
        
        Args:
            image: PIL Image
            prompt: Text prompt
            
        Returns:
            Embedding tensor (2176 dims) or None if failed
        """
        try:
            # Process inputs
            inputs = self.openvla_processor(
                prompt,
                image,
                return_tensors="pt"
            ).to(self.config.device, dtype=torch.bfloat16)
            
            with torch.no_grad():
                # Extract pixel values
                pixel_values = inputs.pixel_values
                
                # Handle 6-channel input (DINO + SigLIP)
                if pixel_values.shape[1] == 6:
                    dino_input = pixel_values[:, :3, :, :]
                    siglip_input = pixel_values[:, 3:, :, :]
                else:
                    dino_input = pixel_values
                    siglip_input = pixel_values
                    
                # Generate DINO features
                dino_features = self.openvla_model.vision_backbone.featurizer(dino_input)
                final_dino_features = dino_features[:, -1, :]  # Last token, shape [1, 1024]
                
                # Generate SigLIP features
                siglip_features = self.openvla_model.vision_backbone.fused_featurizer(siglip_input)
                final_siglip_features = siglip_features.mean(dim=1)  # Average pooling, shape [1, 1152]
                
                # Concatenate features
                concatenated_features = torch.cat(
                    (final_dino_features, final_siglip_features),
                    dim=-1
                )  # Shape [1, 2176]
                
            return concatenated_features
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return None
        
    def _encode_tensor_to_base64(self, tensor: torch.Tensor) -> str:
        """
        Encode PyTorch tensor to base64 string.
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            Base64-encoded string
        """
        buffer = BytesIO()
        torch.save(tensor.cpu(), buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
        
    async def _load_image(self, file: UploadFile) -> Image.Image:
        """
        Load image from uploaded file.
        
        Args:
            file: Uploaded file
            
        Returns:
            PIL Image
        """
        contents = await file.read()
        return Image.open(BytesIO(contents)).convert("RGB")
        
    def _get_health_status(self) -> HealthResponse:
        """
        Get server health status.
        
        Returns:
            HealthResponse with status information
        """
        # Check model status
        models_loaded = {
            "openvla": hasattr(self, 'openvla_model') and self.openvla_model is not None
        }
        
        # Get memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.config.device) / 1e9
            memory_reserved = torch.cuda.memory_reserved(self.config.device) / 1e9
        else:
            memory_allocated = 0
            memory_reserved = 0
            
        memory_usage = {
            "allocated_gb": memory_allocated,
            "reserved_gb": memory_reserved
        }
        
        return HealthResponse(
            status="healthy" if all(models_loaded.values()) else "degraded",
            models_loaded=models_loaded,
            device=self.config.device,
            memory_usage=memory_usage
        )
        
    def run(self):
        """Run the FastAPI server"""
        self.logger.info(f"Starting mix embedding server on {self.config.host}:{self.config.port}")
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            log_level=self.config.log_level.lower()
        )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run the RT-Cache Mix Embedding Server (two-view input)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9021,
        help="Server port (default: 9021, different from single-view server)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, cpu)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Create configuration
    config = ServerConfig()
    config.host = args.host
    config.port = args.port
    config.device = args.device
    config.workers = args.workers
    config.log_level = args.log_level
    
    # Create and run server
    server = MixEmbeddingServer(config)
    server.run()


if __name__ == "__main__":
    main()
