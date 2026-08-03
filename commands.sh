#!/bin/bash
# =============================================================================
# silico/ — all launch commands
# Run from: silico/
# GPU:      export CUDA_VISIBLE_DEVICES=0
# =============================================================================

# export CUDA_VISIBLE_DEVICES=0


# =============================================================================
# SOURCE ONLY (baseline — no adaptation)
# =============================================================================
python train.py --backbone PARENet        --dataset P2PSilico --method Source_Only
python train.py --backbone PARENet        --dataset P2ILReg --method Source_Only


python test.py --backbone PARENet        --dataset P2PSilico --method Source_Only
python test.py --backbone PARENet        --dataset P2ILReg --method Source_Only


# =============================================================================
# LN_TTA
# =============================================================================

# --- test ---
python test.py --backbone PARENet        --dataset P2PSilico --method LN_TTA
python test.py --backbone PARENet        --dataset P2ILReg --method LN_TTA

# --- evaluate (generate_stats) ---
python evaluate.py --backbone PARENet        --dataset P2PSilico --method LN_TTA
python evaluate.py --backbone PARENet        --dataset P2ILReg --method LN_TTA


# =============================================================================
# PEA_TTA
# =============================================================================

# --- test ---
python test.py --backbone PARENet        --dataset P2PSilico --method PEA_TTA
python test.py --backbone PARENet        --dataset P2ILReg --method PEA_TTA

# --- evaluate (generate_stats) ---
python evaluate.py --backbone PARENet        --dataset P2PSilico --method PEA_TTA
python evaluate.py --backbone PARENet        --dataset P2ILReg --method PEA_TTA


# =============================================================================
# Point_TTA  (no generate_stats.py — training via meta_trainval.py)
# =============================================================================

# --- train ---
python train.py --backbone PARENet        --dataset P2PSilico --method Point_TTA --train_phase joint
python train.py --backbone PARENet        --dataset P2PSilico --method Point_TTA --train_phase meta

# --- test ---
python test.py --backbone PARENet        --dataset P2PSilico --method Point_TTA


# =============================================================================
# Purge_Gate
# =============================================================================

# --- test ---
python test.py --backbone PARENet        --dataset P2PSilico --method Purge_Gate
python test.py --backbone PARENet        --dataset P2ILReg --method Purge_Gate

# --- evaluate ---
python evaluate.py --backbone PARENet        --dataset P2PSilico --method Purge_Gate
python evaluate.py --backbone PARENet        --dataset P2ILReg --method Purge_Gate

