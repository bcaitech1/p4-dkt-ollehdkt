var svgWidth = 320;	// SVG요소의 넓이
var svgHeight = 240;	// SVG요소의 높이
var yAxisHeight = svgHeight - 30;	// 가로축의 넓이
var xAxisWidth = svgWidth - 40;	// 세로축의 넓이
var offsetX = 30;	// 가로 오프셋
var offsetY = 10;	// 세로 오프셋
var stepX = 10;	// 막대그래프의 개수. 10 간격.
var xScale;	// 가로 스케일
var yScale;	// 세로 스케일
// 데이터셋
var dataSet = [
		50, 95, 60, 44, 60, 50, 35, 20, 10, 8,
		56, 70, 65, 42, 22, 33, 40, 53, 52, 89,
		90, 55, 50, 55, 65, 72, 45, 35, 15, 45
];
// 히스토그램 설정
var histogram = d3.layout.histogram()
  .range([0, 100])
  .bins(d3.range(0, 100.1, stepX))	// 범위를 계산하여 구함
// 데이터셋으로부터 스케일 계산
function calcScale(){
	// 데이터셋으로부터 최댓값을 구함
	var maxValue = d3.max(histogram(dataSet), function(d, i){
			return d.y;	// 데이터 자체가 아닌 최대 횟수를 반환
		})
	// 세로 스케일 설정
	yScale = d3.scale.linear()
	  .domain([0, maxValue])
	  .range([yAxisHeight, 0])
	// 가로 스케일을 설정
	xScale = d3.scale.linear()
	  .domain([0, 100])
	  .range([0, xAxisWidth ])
}
// 눈금 표시
function drawScale(){
	// 세로 눈금 표시
	d3.select("#myGraph")	// SVG 요소 지정
		  .append("g")	// g요소를 추가. 이것이 눈금을 표시하는 요소가 됨
		  .attr("class", "axis")	// CSS 클래스 지정
		  .attr("transform", "translate("+offsetX+", "+offsetY+")")
		  .call(
				d3.svg.axis()
			  .scale(yScale)  // 스케일 적용
			  .orient("left") //눈금의 표시 위치를 왼쪽으로 지정
			)
	// 가로 눈금 표시
	d3.select("#myGraph")	// SVG 요소 지정
		  .append("g")	// g요소를 추가. 이것이 눈금을 표시하는 요소가 됨
		  .attr("class", "axis")	// CSS 클래스 지정
		  .attr("transform", "translate("+offsetX+", "+(yAxisHeight + offsetY)+")")
		  .call(
				d3.svg.axis()
			  .scale(xScale)  // 스케일 적용
			  .orient("bottom") // 눈금의 표시 위치를 아래쪽으로 지정
			)
}
// 히스토그램 요소 설정
function drawHistgram(){
	var barElements = d3.select("#myGraph")
	  .selectAll("rect")	// rect 요소로 히스토그램 표시
	  .data(histogram(dataSet))	// 데이터를 대응시킴
	  .enter()
	  .append("rect")	// rect 요소를 추가
	  .attr("class", "bar")	// CSS 클래스 추가
	  .attr("width", function(d, i){	// 넓이 설정
			return xScale(d.dx);
		})
	  .attr("x", function(d, i){	// X 좌표 설정
			return i * xScale(d.dx) + offsetX;
		})
	  .attr("y", function(d, i){	// Y 좌표 설정
			return yScale(d.y) + offsetY;
		})
	  .attr("y", yAxisHeight + offsetY)
	  .attr("height", 0)
	  .transition()
	  .duration(1000)
	  .attr("y", function(d, i){	// Y 좌표 설정
			return yScale(d.y) + offsetY;
		})
	  .attr("height", function(d, i){	// 높이 설정
			return yAxisHeight - yScale(d.y);
		})
}
// 간격이 변경되면 히스토그램을 갱신
d3.select("#step").on("change", function(){
	stepX = this.value; 
	histogram
	  .bins(d3.range(0, 100.1, stepX))
	d3.select("#myGraph").selectAll("*").remove();
	calcScale();
	drawHistgram();
	drawScale();
})
// 최초의 히스토그램 표시 처리
calcScale();
drawHistgram();
drawScale();

