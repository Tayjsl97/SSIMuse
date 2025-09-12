# SSIMuse
This is the official implementation of SSIMuse, the first adaptation of the Structural Similarity Index Measure (SSIM) from images to symbolic music to detect data replication. For binary and velocity-based piano rolls, SSIM’s components are reinterpreted and modified, producing SSIMuse-B and SSIMuse-V to evaluate composition and performance dynamics similarity, respectively. 

The musical reinterpretations of comparison functions in SSIMuse-B and SSIMuse-V are shown in the following table.
| Comparison functions | Reinterpretation in SSIMuse-B      | Reinterpretation in SSIMuse-V                  |
|---------------------|-----------------------------------|-----------------------------------------------|
| $l(\mathbf{x},\mathbf{y})$ | Note density consistency          | Overall dynamic consistency                    |
| $c(\mathbf{x},\mathbf{y})$ | ✗ (Invalid)                     | Dynamic dispersion consistency                 |
| $s(\mathbf{x},\mathbf{y})$ | Note event synchronization        | Temporal pattern consistency of velocity changes |
