var svgWidth = 320;	// SVG요소의 넓이
var svgHeight = 240;	// SVG요소의 높이
// 데이터셋
var dataSet = {
	children : [
		{ value : 10 },
		{ value : 8 },
		{ value : 2 }
	]
}
// 트리맵 레이아웃
var treemap = d3.layout.treemap()
  .size([svgWidth, svgHeight])	// SVG 요소의 넓이에 맞춤
// 트리맵 그리기
var tmap = d3.select("#myGraph")
  .selectAll("rect")	// rect 요소 지정
  .data(treemap.nodes(dataSet))	// 노드를 대상으로 처리
// 분할 맵 영역 추가
tmap.enter()
  .append("rect")	// rect 요소를 추가
  .attr("class", "block")	// CSS 클래스 추가
  .attr("x", function(d, i) {	// X 좌표 설정
		return d.x;
	}) 
  .attr("y", function(d, i) {	// Y 좌표 설정
		return d.y;
	})
  .attr("width", function(d, i) {	// 넓이 설정
		return d.dx;
	})
  .attr("height", function(d, i) {	// 높이 설정
		return d.dy;
	})