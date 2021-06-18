var svgWidth = 320;	// SVG요소의 넓이
var svgHeight = 240;	// SVG요소의 높이
// 데이터셋
var dataSet = {
	children: [
		{ value : 100 },
		{ children: [
				{ value : 50 },
				{ children: [
						{ value: 30 },
						{ value: 20 },
						{ value: 10 }
					]
				}
			]
		}
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
  .append("rect")  // rect 요소를 추가
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
  .style("opacity", function(d, i){	// 깊이에 따라 투명도를 설정
		return d.depth / 5;	// 자식이 깊어지면 짙어짐
	})
// 맵 안에 문자 추가
tmap.enter()
  .append("text")	// text 요소를 추가
  .attr("class", "name")	// CSS 클래스 추가
  .attr("transform", function(d, i){	// 위치를 계산하여 XY 좌표를 일괄 설정
		return "translate(" + (d.x+d.dx/2) + "," + (d.y+d.dy/2)+")";	
	})
  .attr("dy", "0.35em")	// 표시 위치 조정
  .text(function(d, i) {  // 문자 표시
		if ((d.depth == 0) || (d.children)) {	// 루트인가 자식 노드가 있는가
			return null;	// 루트이거나 자식 노드가 있다면 null을 반환하여 아무것도 표시하지 않음 
		}
		return d.value;	// 영역 안에 표시할 문자를 반환
	})
	