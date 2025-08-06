# AWS EKS Cluster configuration for Cyber-LLM
# Terraform configuration for AWS deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "Cyber-LLM"
      Environment = var.environment
      Owner       = "cyber-llm-team"
      ManagedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_caller_identity" "current" {}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "cyber-llm-vpc-${var.environment}"
  cidr = var.vpc_cidr
  
  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true
  
  # EKS specific tags
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }
  
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "cyber-llm-${var.environment}"
  cluster_version = var.kubernetes_version
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Cluster endpoint configuration
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access_cidrs = var.allowed_cidr_blocks
  
  # Encryption configuration
  cluster_encryption_config = [
    {
      provider_key_arn = aws_kms_key.eks.arn
      resources        = ["secrets"]
    }
  ]
  
  # EKS Managed Node Groups
  eks_managed_node_groups = {
    # CPU-optimized nodes for general workloads
    cpu_nodes = {
      name = "cpu-nodes"
      
      instance_types = ["c5.2xlarge"]
      min_size      = 2
      max_size      = 10
      desired_size  = 3
      
      labels = {
        role = "cpu-worker"
      }
      
      taints = []
    }
    
    # GPU nodes for AI/ML workloads
    gpu_nodes = {
      name = "gpu-nodes"
      
      instance_types = ["p3.2xlarge", "g4dn.2xlarge"]
      min_size      = 1
      max_size      = 5
      desired_size  = 2
      
      labels = {
        role = "gpu-worker"
      }
      
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  # Fargate profiles for serverless workloads
  fargate_profiles = {
    cyber_llm_fargate = {
      name = "cyber-llm-fargate"
      selectors = [
        {
          namespace = "cyber-llm"
          labels = {
            compute-type = "fargate"
          }
        }
      ]
    }
  }
  
  # OIDC Identity provider
  cluster_identity_providers = {
    sts = {
      client_id = "sts.amazonaws.com"
    }
  }
}

# KMS key for EKS encryption
resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key for Cyber-LLM"
  deletion_window_in_days = 7
  enable_key_rotation     = true
}

resource "aws_kms_alias" "eks" {
  name          = "alias/eks-cyber-llm-${var.environment}"
  target_key_id = aws_kms_key.eks.key_id
}

# ECR Repository for container images
resource "aws_ecr_repository" "cyber_llm" {
  name                 = "cyber-llm"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  encryption_configuration {
    encryption_type = "KMS"
    kms_key        = aws_kms_key.ecr.arn
  }
}

resource "aws_kms_key" "ecr" {
  description             = "ECR Encryption Key for Cyber-LLM"
  deletion_window_in_days = 7
  enable_key_rotation     = true
}

# S3 bucket for model artifacts
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "cyber-llm-models-${var.environment}-${random_string.bucket_suffix.result}"
}

resource "aws_s3_bucket_encryption_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_kms_key" "s3" {
  description             = "S3 Encryption Key for Cyber-LLM"
  deletion_window_in_days = 7
  enable_key_rotation     = true
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# RDS PostgreSQL for application data
resource "aws_db_subnet_group" "cyber_llm" {
  name       = "cyber-llm-${var.environment}"
  subnet_ids = module.vpc.database_subnets
  
  tags = {
    Name = "cyber-llm-${var.environment}"
  }
}

resource "aws_security_group" "rds" {
  name_prefix = "cyber-llm-rds-${var.environment}"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "cyber_llm" {
  allocated_storage           = var.db_allocated_storage
  max_allocated_storage      = var.db_max_allocated_storage
  storage_type               = "gp3"
  storage_encrypted          = true
  kms_key_id                = aws_kms_key.rds.arn
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class
  
  identifier = "cyber-llm-${var.environment}"
  db_name    = "cyber_llm"
  username   = var.db_username
  password   = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.cyber_llm.name
  
  backup_retention_period = 7
  backup_window          = "07:00-09:00"
  maintenance_window     = "sun:09:00-sun:10:00"
  
  skip_final_snapshot = var.environment == "dev" ? true : false
  deletion_protection = var.environment == "prod" ? true : false
  
  performance_insights_enabled = true
  monitoring_interval         = 60
}

resource "aws_kms_key" "rds" {
  description             = "RDS Encryption Key for Cyber-LLM"
  deletion_window_in_days = 7
  enable_key_rotation     = true
}

# ElastiCache Redis for caching
resource "aws_elasticache_subnet_group" "cyber_llm" {
  name       = "cyber-llm-${var.environment}"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name_prefix = "cyber-llm-redis-${var.environment}"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
}

resource "aws_elasticache_replication_group" "cyber_llm" {
  replication_group_id         = "cyber-llm-${var.environment}"
  description                  = "Redis cluster for Cyber-LLM"
  
  node_type            = var.redis_node_type
  port                 = 6379
  parameter_group_name = "default.redis7"
  
  num_cache_clusters = var.redis_num_cache_nodes
  
  subnet_group_name  = aws_elasticache_subnet_group.cyber_llm.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                = var.redis_auth_token
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  snapshot_retention_limit = 7
  snapshot_window         = "07:00-09:00"
  
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow.name
    destination_type = "cloudwatch-logs"
    log_format       = "text"
    log_type         = "slow-log"
  }
}

resource "aws_cloudwatch_log_group" "redis_slow" {
  name              = "/aws/elasticache/cyber-llm-${var.environment}/slow-log"
  retention_in_days = 7
}
