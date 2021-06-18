var svgWidth = 320;	// SVG 요소의 넓이
var svgHeight = 240;	// SVG 요소의 높이
var dataSet = [10, 47, 65, 8, 64, 99, 75, 22, 63, 80];	// 데이터셋
var margin = svgWidth/(dataSet.length - 1);	// 꺾은선 그래프의 간격 계산
// 꺾은선 그래프의 좌표를 계산하는 메서드
var line = d3.svg.line()	// svg의 선
	.x(function(d, i){
		return i * margin;	// X 좌표는 표시 순서×간격
	})
	.y(function(d, i){
		return svgHeight - d;	// 데이터로부터 Y 좌표 빼기
	})
// 꺾은선 그래프 그리기
var lineElements = d3.select("#myGraph")
	.append("path")	// 데이터 수만큼 path 요소가 추가됨
	.attr("class", "line")	// CSS 클래스 지정
	.attr("d", line(dataSet))	//연속선 지정
