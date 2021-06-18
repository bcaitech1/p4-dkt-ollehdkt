var svgWidth = 320;	// SVG요소의 넓이
var svgHeight = 240;	// SVG요소의 높이
var barWidth = svgWidth / 11;	// 막대의 넓이
// 데이터셋
var dataSet = [
		50, 95, 60, 44, 60, 50, 35, 20, 10, 8,
		56, 70, 65, 42, 22, 33, 40, 53, 52, 89,
		90, 55, 50, 55, 65, 72, 45, 35, 15, 45
];
// 히스토그램 데이터를 설정
var histogram = d3.layout.histogram()
  .range([0, 100])
  .bins([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
// 데이터셋으로부터 최댓값을 구함
var maxValue = d3.max(histogram(dataSet), function(d, i){
		return d.y;	// 데이터 자체가 아닌 최대 횟수를 반환
	})
// 세로 스케일 설정
var yScale = d3.scale.linear()
  .domain([0, maxValue])
  .range([svgHeight, 0])
// 히스토그램 그리기
var barElements = d3.select("#myGraph")
  .selectAll("rect")	// rect 요소로 히스토그램 표시
  .data(histogram(dataSet))	// 데이터를 대응시킴
  .enter()
  .append("rect")	// rect 요소를 추가
  .attr("class", "bar")	// CSS 클래스 추가
  .attr("x", function(d, i){	// X 좌표 설정
		return i * barWidth;
	})
  .attr("y", function(d, i){	// Y 좌표 설정
		return yScale(d.y);
	})
  .attr("width", barWidth)	// 넓이 설정
  .attr("height", function(d, i){	// 높이 설정
		return svgHeight - yScale(d.y);
	})