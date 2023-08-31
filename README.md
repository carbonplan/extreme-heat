<p align="left" >
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://carbonplan-assets.s3.amazonaws.com/monogram/light-small.png">
  <img alt="CarbonPlan monogram." height="48" src="https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png">
</picture>
</p>

# carbonplan / extreme heat

## context

This repository contains code to estimate wet-bulb globe temperatures (WBGT) in the shade and in the sun through 2060, developed as part of a [collaborative project with _The Washington Post_](TK). We wrote an [explainer article](https://carbonplan.org/research/extreme-heat-explainer) that describes our methods and the resulting dataset, as well as a [blog post](https://carbonplan.org/blog/open-risk-data) on the importance of public access and transparency in climate data and services. To support reproducibility and further research, the code here is sufficient to run the entire analysis. The [notebooks folder](/notebooks) contains 10 Jupyter notebooks that, in sequence, show how we created our dataset, including the inputs and algorithms we used and all the assumptions that we made.

## data

The final output dataset includes historical and future estimates of WBGT in the shade and in the sun for ~15,300 cities and ~24,000 climatically-similar regions around the world. The full analysis includes 26 global climate models (GCMs) and one emissions scenario (SSP2-4.5). To support different use cases, we've made the results available at a few different levels of granularity and in two different formats. The available data is summarized below in order of increasing detail and size. For each row, click on the `List` to see a full list of files and their associated URLs.

| Data                                            | Format | Files                                   | Uses                                                     |
| ----------------------------------------------- | ------ | --------------------------------------- | -------------------------------------------------------- |
| Medians over time and medians over GCMs         | CSV    | [List](/data/csv_locations.md)          | Common summary statistics of extreme heat                |
| Medians over time for the full ensemble of GCMs | Zarr   | [List](/data/zarr_summary_locations.md) | Inspecting variability of summary statistics across GCMs |
| Daily projections for the full ensemble of GCMs | Zarr   | [List](/data/zarr_daily_locations.md)   | Inspecting full daily time series across GCMs            |

## about us

CarbonPlan is a non-profit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions with open data and tools. [Find out more at carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/extreme-heat/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
