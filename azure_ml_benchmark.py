#!/usr/bin/env python3
"""
Azure ML Compute approach for running FoldForever benchmark on premium GPUs.
This bypasses AKS quota limitations by using Azure ML's compute instances.
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, Command, Data
from azure.identity import DefaultAzureCredential
import os

def create_ml_benchmark_job():
    """Create Azure ML job for FoldForever benchmark on premium GPUs."""
    
    # Initialize ML Client
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id="39bd5d25-e94b-4a97-a192-f0781446d526",  # Your subscription
        resource_group_name="protein-folding",
        workspace_name="protein-folding-ml"  # You'll need to create this
    )
    
    # Define environment with all dependencies
    environment = Environment(
        name="foldforever-benchmark-env",
        description="Environment for FoldForever protein folding benchmark",
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/pytorch-1.13-ubuntu20.04-py38-cuda11.7-gpu:latest"
    )
    
    # Define the benchmark command
    command_job = Command(
        code="./",  # Local directory with benchmark code
        command="python benchmark_casp14_foldforever_vs_baselines.py --mode full --gpu --sequences 30 --output /tmp/results --verbose",
        environment=environment,
        compute="gpu-cluster-a100",  # Premium GPU cluster
        display_name="FoldForever-Benchmark-A100",
        description="Comprehensive FoldForever benchmark on A100 GPUs with real CASP14/CAMEO datasets",
        tags={"project": "foldforever", "type": "benchmark", "gpu": "a100"}
    )
    
    # Submit the job
    job = ml_client.jobs.create_or_update(command_job)
    print(f"Job submitted: {job.name}")
    print(f"Job URL: {job.studio_url}")
    
    return job

if __name__ == "__main__":
    create_ml_benchmark_job()
