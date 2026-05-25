# MTRefSeg Project Page

Static project page for **MTRefSeg: An Open-Source Benchmark and Baseline for Multi-temporal Referring Segmentation**.

This version follows the uploaded `MTRefSeg.pdf`: the paper link points to `assets/MTRefSeg.pdf`, the author list follows the PDF, and the Stage-1 resource is labeled as **MT-Stage1** with approximately **20K** vision-only bi-temporal samples.

## Preview

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Files

- `index.html`: project page
- `assets/MTRefSeg.pdf`: paper PDF
- `assets/pic_1.png`: motivation and overview
- `assets/pic_task_intro.png`: task comparison
- `assets/pic_dataset.png`: dataset examples
- `assets/pic_table_dataset.png`: dataset statistics and multi-domain comparison
- `assets/pic_dataset_generate.png`: CRAFT-Agent construction pipeline
- `assets/pic_method.png`: MTRefSeg-R1 framework
- `assets/pic_adapt_models.png`: adapting VLM/LVLM models to MTRS
- `assets/pic_table_trainval.png`: Train→Val quantitative results
- `assets/pic_table_ns.png`: NS-domain quantitative results
- `assets/pic_table_rs.png`: RS-domain quantitative results
- `assets/pic_vis_NS.png`: normal-scene qualitative results
- `assets/pic_vis_RS.png`: remote-sensing qualitative results
- `assets/pic_attention.png`: language-guided temporal attention visualization
