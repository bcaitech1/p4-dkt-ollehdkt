var svgWidth = 320;	// SVG 요소의 넓이
var svgHeight = 240;	// SVG 요소의 높이
var offsetX = 30;	// 가로 오프셋
var offsetY = 20;	// 세로 오프셋
var scale = 2.0;	// 2배 크기로 그리기
var dataSet1 = [10, 47, 65, 8, 64, 99, 75, 22, 63, 80];	// 데이터셋1
var dataSet2 = [90, 77, 55, 48, 64, 90, 85, 42, 13, 40];	// 데이터셋2
var dataSet3 = [50, 27, 45, 58, 84, 70, 45, 22, 30, 90];	// 데이터셋3
var margin = svgWidth /(dataSet1.length - 1);	// 꺾은선 그래프의 간격 계산
drawGraph(dataSet1, "itemA");	// itemA의 꺾은선 그래프 표시
drawGraph(dataSet2, "itemB");	// itemB의 꺾은선 그래프 표시
drawGraph(dataSet3, "itemC");	// itemB의 꺾은선 그래프 표시
drawScale();	// 눈금 표시
// 꺾은선 그래프를 표시하는 함수
function drawGraph(dataSet, cssClassName){
	// 꺾은선 그래프의 좌표를 계산하는 메서드
	var line = d3.svg.line()	// svg의 선
	  .x(function(d, i){
			return offsetX + i * margin;	// X 좌표는 표시 순서×간격
		})
	  .y(function(d, i){
			return svgHeight - (d * scale) - offsetY;	// 데이터로부터 Y 좌표 빼기
		})
	// 꺾은선 그래프 그리기
	var lineElements = d3.select("#myGraph")
	  .append("path")	// 데이터 수만큼 path 요소가 추가됨
	  .attr("class", "line "+cssClassName)	// CSS 클래스 지정
	  .attr("d", line(dataSet))	//연속선 지정
}
// 그래프의 눈금을 표시하는 함수
function drawScale(){
	// 눈금을 표시하기 위해 D3 스케일 설정
	var yScale = d3.scale.linear()  // 스케일 설정
	  .domain([0, 100])   // 원래 크기
	  .range([scale*100, 0]) // 실제 표시 크기
	// 눈금 표시
	d3.select("#myGraph")	// SVG 요소를 지정
		  .append("g")	// g 요소 추가. 이것이 눈금을 표시하는 요소가 됨
		  .attr("class", "axis")	// CSS 클래스 지정
		  .attr("transform", "translate("+offsetX+", "+offsetY+")")
		  .call(
				d3.svg.axis()
			  .scale(yScale)  //스케일 적용
			  .orient("left") //눈금 표시 위치를 왼쪽으로 지정
			)
		// 가로 방향의 선을 표시
		d3.select("#myGraph")
		  .append("rect")	// rect 요소 추가
		  .attr("class", "axis_x")	// CSS 클래스 지정
		  .attr("width", svgWidth)	// 선의 굵기를 지정
		  .attr("height", 1)	// 선의 높이를 지정
		  .attr("transform", "translate("+offsetX+", "+(svgHeight-offsetY-0.5)+")")
}
