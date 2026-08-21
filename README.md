# NFF — Normalized Factor Flow

<p align="center">
  <strong>A framework for structural analysis of continuous data flows.</strong>
</p>

<p align="center">
  <a href="./DOCUMENTATION.md">Documentation</a>
  ·
  <a href="#installation">Installation</a>
  ·
  <a href="#usage">Usage</a>
</p>

---

## About

**NFF (Normalized Factor Flow)** transforms data flows into normalized structural representations.

It analyzes temporal **Containers** through multiple **NUA** units, produces an **NFF Signature**, and aggregates the result into an **ORDER** value.

NFF is designed to be domain-independent and suitable for both static and real-time data.

---

## Installation

### Windows

```powershell
iwr -useb https://raw.githubusercontent.com/Elxrth/NFF-normalized-factor-flow/main/install.ps1 | iex
```

### Manual

```bash
git clone https://github.com/Elxrth/NFF-normalized-factor-flow.git
cd NFF-normalized-factor-flow
```

---

## Usage

A typical NFF analysis follows:

```text
Flow → Containers → NUA → Normalization → Signature → ORDER
```

For real-time analysis, the resulting `ORDER(t)` sequence can be treated as a new Flow.

---

## Documentation

For the complete specification and theoretical background:

**[Read the documentation →](./DOCUMENTATION.md)**

---

## Status

NFF is an evolving framework. Results depend on the selected NUA units, normalization method, Container Delay, sampling frequency, and weighting strategy.

---

<p align="center">
  <sub>NFF — Normalized Factor Flow · Elxrth</sub>
</p>
