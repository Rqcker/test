import { covidData } from "./dataControl.js";

// create listener and new model
const listeners = [];
const model = {
  selectedCountry: 'OWID_WRL',
  axisValue: 'date',
  brushedValue: null,
  brushedBounds: null,
  bounds: null,
  mapColors: null,
  brushedCountries: null,
  hoveredCountry: null,
};

// update the data model
export function updateModel(changes) {
  const keys = {};

  for (const key in changes) {
    if (key in model) {
      if (model[key] != changes[key]) {
        model[key] = changes[key];
        keys[key] = changes[key];
      }
    }
  }
  for (const list of listeners) {
    list(keys);
  }
}

const width = 500
const height = 250
const margin = { top: 10, right: 30, bottom: 50, left: 70 };
const innerWidth = width - margin.left - margin.right;
const innerHeight = height - margin.top - margin.bottom;

class LineChart {
  // Holds current array of data records
  lineData;

  // Attribute on each axis
  vertical;
  horizontal;
  timeline; // Whether horizontal axis is a timeline
  bounds; // Zooming bounds on horizontal

  // Display elements accesed and updated across methods
  xAxis;
  yAxis;
  selectBox;
  line;
  focusCircle;
  focusText;

  // Scales may update when country changes or axes change
  xScale;
  yScale;

  constructor(elementId, defaultStat) {
    const container = d3
      .select(`#${elementId}`)
      .append("div")
      .classed("line-chart-container", true);

    // Allow changing the chart display
    this.vertical = defaultStat;
    container
      .append("select")
      .on("change", this.onVerticalSelect.bind(this))
      .selectAll("option")
      .data(covidData.toPlot)
      .join("option")
      .attr("value", (d) => d)
      .property("selected", (d) => d === defaultStat)
      .text((d) => d.split("_").join(" "));

    const chart = container
      .append("svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .classed("svg-line-chart", true)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Scales are reused across event listeners
    this.yScale = d3.scaleLinear().range([innerHeight, 0]);

    this.xAxis = chart
      .append("g")
      .attr("transform", `translate(0, ${innerHeight})`);

    this.yAxis = chart.append("g");

    this.selectBox = chart
      .append("rect")
      .classed("line-chart-select", true)
      .attr("width", 0)
      .attr("height", innerHeight)
      .attr("x", 0)
      .attr("y", 0);

    this.line = chart
      .append("path")
      .classed("line-line", true)
      .classed(`line-${defaultStat}`, true);

    // Circle will show on mouseover, hidden initially
    this.focusCircle = chart
      .append("circle")
      .classed("line-highlight", true)
      .attr("r", 8.5);

    // Text of value appears on hover, above the line
    this.focusText = chart
      .append("text")
      .classed("line-hint", true)
      .attr("text-anchor", "end")
      .attr("alignment-baseline", "middle");

    addModelListener(this.onModelUpdate.bind(this));

    chart
      .append("rect")
      .attr("fill", "none")
      .attr("pointer-events", "all")
      .attr("width", innerWidth)
      .attr("height", innerHeight)
      .on("mousedown", this.onMouseDown.bind(this))
      .on("mousemove", this.onMouseMove.bind(this))
      .on("mouseup", this.onMouseUp.bind(this))
      .on("mouseout", this.onMouseOut.bind(this));
  }

  get domain() {
    return d3.extent(this.lineData, (d) => d[this.horizontal]);
  }

  updateHorizontalAxis() {
    // Only date dimension is non-numeric
    this.xScale = this.timeline ? d3.scaleTime() : d3.scaleLinear();
    this.xScale.range([0, innerWidth]);

    // By default, use full extent of horizontal axis to reflect missing data
    this.xScale.domain(this.bounds ?? this.domain);

    const x = d3.axisBottom(this.xScale);

    // Prevent large numeric values from creating long tick labels
    if (!this.timeline) {
      x.ticks(null, "s");
    }

    this.xAxis.transition().duration(2000).call(x);

    this.updateLine();
  }

  updateVerticalAxis() {
    // Quantity of interest should always be shown with respect to 0
    this.yScale.domain([0, d3.max(this.lineData, (d) => d[this.vertical])]);
    this.yAxis
      .transition()
      .duration(2000)
      .call(d3.axisLeft(this.yScale).ticks(null, "s"));

    this.updateLine();
  }

  updateLine() {
    // Not all records will have both data dimensions present
    // Compare to null to keep values of 0
    this.line.datum(
      this.lineData.filter(
        (d) => d[this.vertical] != null && d[this.horizontal] != null
      )
    );

    // Collapse existing line first to avoid density change artifacts
    // (some still occur, but not as bad)
    this.line
      .transition()
      .duration(1000)
      .attr(
        "d",
        d3
          .line()
          .x((d) => this.xScale(d[this.horizontal]))
          .y(() => this.yScale(this.yScale.domain()[0]))
      )
      .transition()
      .duration(1000)
      .attr(
        "d",
        d3
          .line()
          .x((d) => this.xScale(d[this.horizontal]))
          .y((d) => this.yScale(d[this.vertical]))
      );
  }

  highlightPoint(brushedValue) {
    const { horizontal, vertical, focusCircle, focusText } = this;

    // Value set to null when mouse leaves chart
    if (!brushedValue) {
      this.showHighlight(false);
      return;
    }

    const data = this.line.datum();

    if (!data.length) {
      this.showHighlight(false);
      return;
    }

    // Only highlight if datapoint exists aligned with cursor
    if (
      brushedValue < data[0][horizontal] ||
      brushedValue > data[data.length - 1][horizontal]
    ) {
      focusText.classed("visible", false);
      focusCircle.classed("visible", false);
      return;
    }

    // Find closest data point to left of brushed time
    const bisect = d3.bisector((d) => d[horizontal]).left;
    const index = bisect(data, brushedValue);
    const datapoint = data[index];

    if (!datapoint) {
      this.showHighlight(false);
      return;
    }

    // Convert back to range coordinate space to position elements
    const x_scaled = this.xScale(datapoint[horizontal]);
    const y_scaled = this.yScale(datapoint[vertical]);

    focusCircle.attr("cx", x_scaled).attr("cy", y_scaled);
    focusText
      .attr("x", x_scaled)
      .attr("y", y_scaled - 15)
      .text(datapoint[vertical].toLocaleString());

    this.showHighlight(true);
  }

  updateChart(hAxis, vAxis) {
    if (hAxis) this.updateHorizontalAxis();
    if (vAxis) this.updateVerticalAxis();
  }

  onVerticalSelect(event) {
    this.vertical = event.target.value;
    this.updateChart(false, true);
  }

  async onModelUpdate(changes) {
    const { axisValue, selectedCountry } = changes;
    let updateV = false;
    let updateH = false;

    if (selectedCountry) {
      const allData = await covidData();
      const countryInfo = allData[selectedCountry];
      this.lineData = countryInfo.data;

      updateV = true;
      updateH = true;
    }

    if (axisValue) {
      this.horizontal = axisValue;
      this.timeline = this.horizontal === "date";

      updateH = true;
    }

    if ("bounds" in changes) {
      this.bounds = changes.bounds;
      updateH = true;
    }

    this.updateChart(updateH, updateV);

    if ("brushedBounds" in changes) {
      if (changes.brushedBounds) {
        const [xl, xr] = changes.brushedBounds;
        this.selectBox.attr("x", xl).attr("width", xr - xl);
      } else {
        this.clearSelection();
      }
    }

    // Brushed value can be null to represent no value
    if ("brushedValue" in changes) {
      this.highlightPoint(changes.brushedValue);
    }
  }

  onMouseDown(event) {
    const [x] = d3.pointer(event);
    this.mouseDown = x;
  }

  onMouseMove(event) {
    if (!this.xScale) return;

    // Need x-value of mouse position in domain coordinate space
    const [x_mouse] = d3.pointer(event);
    const x_value = this.xScale.invert(x_mouse);

    updateModel({ brushedValue: x_value });

    // Value could be 0
    if (this.mouseDown != null) {
      const [x] = d3.pointer(event);
      updateModel({
        brushedBounds: d3.extent([this.mouseDown, x]),
      });
    }
  }

  onMouseUp(event) {
    const [x] = d3.pointer(event);
    const bounds = d3.extent([this.mouseDown, x], (d) => this.xScale.invert(d));

    // Prevent zooming below 10% of data domain (prevents zoom on click)
    const [lower, upper] = this.domain;
    const tooSmall = bounds[1] - bounds[0] < (upper - lower) * 0.1;

    updateModel({
      brushedBounds: null,
      bounds: tooSmall ? this.bounds : bounds,
    });
  }

  onMouseOut() {
    updateModel({
      brushedValue: null,
      brushedBounds: null,
    });
  }

  clearSelection() {
    this.mouseDown = null;
    this.selectBox.attr("width", 0);
  }

  showHighlight(show) {
    this.focusText.classed("visible", show);
    this.focusCircle.classed("visible", show);
  }
}

class ClusterChart {
  // Will store coordinates when brushing started
  mouseDown;

  constructor(elementId, x, y) {
    this.x = x;
    this.y = y;

    const container = d3.select(`#${elementId}`)
      .append('div')
      .classed('cluster-chart', true);

    const svg = container.append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .classed('svg-clustered', true);

    this.chart = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales are reused across event listeners
    this.xScale = d3.scaleLinear()
      .range([0, innerWidth]);

    this.yScale = d3.scaleLinear()
      .range([innerHeight, 0]);

    this.xAxis = this.chart.append('g')
      .attr('transform', `translate(0, ${innerHeight})`);

    this.yAxis = this.chart.append('g');

    this.selectBox = this.chart.append('rect')
      .classed('scatter-box', true);

    // Add an x-axis label under the chart
    svg.append('text')
      .text(x.split('_').join(' '))
      .attr('text-anchor', 'middle')
      .attr('x', margin.left + innerWidth / 2)
      .attr('y', height - 5);

    // Add a y-axis label to the left
    svg.append('text')
      .text(y.split('_').join(' '))
      .attr('text-anchor', 'middle')
      .attr('transform', `translate(20,${margin.top + innerHeight / 2}) rotate(-90)`);
  }

  // Performs k-means clustering and updates the chart
  async cluster(numClusters) {
    const data = await covidData();

    const vectors = Object.keys(data).map(k => {
      // Ignore special entries, they're not comparable
      if (k.startsWith('OWID')) return null;

      // Taking copy to order data from newest to oldest non-destructively
      const allStats = data[k].data.slice().reverse();
      // Need to find latest statistic with both attributes present
      const latestIndex = allStats.findIndex(d => {
        return (
          // Compare to null to allow for values of 0
          d[this.x] != null
          && d[this.y] != null
        )
      })
      const latestStats = allStats[latestIndex];

      // Some countries may not have information available
      if (!latestStats) return null;

      return {
        label: data[k].location,
        x: latestStats[this.x],
        y: latestStats[this.y],
      };
    }).filter(d => d);

    // ==== Perform the k-means algorithm ====
    // First need k initial means (using Forgy method)
    const means = d3.range(numClusters).map(i => ({ ...vectors[i] }));

    // Repeate until centroids converge
    let meansMoved = true;
    while (meansMoved) {
      // Now assign each vector the nearest mean
      vectors.forEach(v => {
        const dists = [];
        for (let i = 0; i < means.length; i++) {
          const { x: mx, y: my } = means[i];
          dists[i] = (v.x - mx) ** 2 + (v.y - my) ** 2
        }

        // Assign the vector to a cluster (identified by index)
        v.cluster = dists.indexOf(Math.min(...dists));
      });

      // Prepare to check if any means have updated
      meansMoved = false;
      const oldMeans = means.slice();

      // Now recalculate the means
      for (let i = 0; i < means.length; i++) {
        const inCluster = vectors.filter(v => v.cluster === i);
        means[i].x = average(inCluster.map(v => v.x));
        means[i].y = average(inCluster.map(v => v.y));

        if (means[i].x !== oldMeans[i].x
          && means[i].y !== oldMeans[i].y
        ) {
          meansMoved = true;
        }
      }
    }

    // Some outliers blow up the scale, ignore them
    this.xScale.domain([0, d3.quantile(vectors, 0.75, d => d.x)]);
    this.yScale.domain(d3.extent(vectors, d => d.y));
    this.colorScale = d3.scaleOrdinal(d3.schemeSet1)
      .domain(d3.range(numClusters));

    this.centroids = means;
    this.vectors = vectors;
    this.update();
  }

  update() {
    this.points = this.chart
      .append('g')
      .selectAll('circle')
      .data(this.vectors)
      .join('circle')
      .attr('cx', d => this.xScale(d.x))
      .attr('cy', d => this.yScale(d.y))
      .attr('r', 2)
      .attr('fill-opacity', 0.8);

    this.xAxis.call(d3.axisBottom(this.xScale).ticks(6, 's'));
    this.yAxis.call(d3.axisLeft(this.yScale).ticks(6, 's'));

    this.chart.append('rect')
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .attr('width', innerWidth)
      .attr('height', innerHeight)
      .on('mousedown', this.onMouseDown.bind(this))
      .on('mouseup', this.onMouseUp.bind(this))
      .on('mousemove', this.onMouseMove.bind(this))
      .on('mouseout', this.onMouseUp.bind(this));

    addModelListener(this.onModelUpdate.bind(this));
  }

  onMouseDown(event) {
    const [x, y] = d3.pointer(event);
    this.mouseDown = [x, y];
    this.selectBox
      .attr('width', 0)
      .attr('height', 0);
    this.selectBox.classed('visible', true);
  }

  onMouseUp() {
    this.mouseDown = null;
    this.selectBox.classed('visible', false);
  }

  onMouseMove(event) {
    if (this.mouseDown) {
      const [x, y] = d3.pointer(event);
      const [mx, my] = this.mouseDown;

      const [xl, xr] = [
        Math.min(x, mx),
        Math.max(x, mx),
      ];
      const [yt, yb] = [
        Math.min(y, my),
        Math.max(y, my),
      ];

      // Box dimensions can't go negative
      this.selectBox
        .attr('x', xl)
        .attr('y', yt)
        .attr('width', xr - xl)
        .attr('height', yb - yt);

      // Need box bounds in original scales to find points within
      const [xlp, xrp] = [
        this.xScale.invert(xl),
        this.xScale.invert(xr),
      ];

      const [ytp, ybp] = [
        this.yScale.invert(yt),
        this.yScale.invert(yb),
      ];

      // The points that lie in the dragged rectangle to are brushed
      const brushed = this.vectors.filter(v => {
        return (
          (v.x > xlp && v.x < xrp)
          && (v.y < ytp && v.y > ybp)
        );
      });

      updateModel({ brushedCountries: brushed.map(v => v.label) });
    }
  }

  onModelUpdate(changes) {
    // Avoid frequent expensive updates when unchanged
    if ('brushedCountries' in changes) {
      const { brushedCountries } = changes;

      this.points.attr('fill', d => {
        if (brushedCountries && brushedCountries.includes(d.label)) {
          return 'white';
        } else {
          return this.colorScale(d.cluster);
        }
      })
    }
  }
}

/**
 * Finds the average of an array of numbers, rounding to 2 decimal places
 * @param {number[]} values array of numbers
 */
function average(values) {
  // Cumulative moving average
  const avg = values.reduce((pv, cv, i) => pv + (cv - pv) / (i + 1), 0);

  // Want to enforce decimal places so clustering converges
  return Number(avg.toFixed(2));
}

export function addModelListener(listener) {
  listeners.push(listener);
  listener(model);
}

// load the Covid19 data
covidData();

// active the tool bar
addModelListener(async ({ selectedCountry }) => {
  if (!selectedCountry) return;

  const data = await covidData();
  d3.select("#charts-title").text(
    `Showing data for ${data[selectedCountry].location}`
  );
});

// can change horizontal axis of all line charts
d3.select("#charts-select")
  .on("change", (event) => {
    updateModel({
      axisValue: event.target.value,
      bounds: null,
    });
  })
  .selectAll("option")
  .data(covidData.toPlotAgainst)
  .join("option")
  .attr("value", (d) => d)
  .text((d) => d.split("_").join(" "));

// clear country button
d3.select("#reset-selection").on("click", () =>
  updateModel({ selectedCountry: "OWID_WRL" })
);

// when the country already selected, the button working
addModelListener((changes) => {
  if ("selectedCountry" in changes) {
    d3.select("#reset-selection").property(
      "disabled",
      changes.selectedCountry === "OWID_WRL"
    );
  }
});

// clear line chart zoom
d3.select("#reset-bounds").on("click", () => updateModel({ bounds: null }));

// when the charts already zoomed, the button working
addModelListener((changes) => {
  if ("bounds" in changes) {
    d3.select("#reset-bounds").property("disabled", !changes.bounds);
  }
});

new LineChart("line-charts", "new_cases_smoothed");
new LineChart("line-charts", "new_deaths_smoothed");
new LineChart("line-charts", "people_vaccinated");
new LineChart("line-charts", "people_fully_vaccinated");

// visualise relation between wealth and pandemic
new ClusterChart(
  "scatter-plots",
  "gdp_per_capita",
  "total_cases_per_million"
).cluster(3);
new ClusterChart(
  "scatter-plots",
  "population_density",
  "total_cases_per_million"
).cluster(3);
