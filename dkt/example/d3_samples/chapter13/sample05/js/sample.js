var svgWidth = 320;	// SVG요소의 넓이
var svgHeight = 240;	// SVG요소의 높이
// 데이터셋
var dataSet = {
	name: "루트",
	children: [
		{ name: "Dir1", children: [
			{ name: "Dir2", children: [
				{ name: "파일 A", value: 5000 },
				{ name: "파일 B", value: 3000 },
				{ name: "Dir3", children: [
					{ name: "파일 C", value: 2000 },
					{ name: "Dir4", children: [
							{ name: "파일 D", value: 1000 },
							{ name: "파일 E", value: 1500 }
					]
					}
				]
				}
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
  .append("rect")	// rect 요소를 추가
  .attr("class", "file")	// CSS 클래스 추가
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
  .style("opacity", 0)	// 영역을 투명에서 불투명으로 함
  .transition()
  .delay(function(d, i){	// 시간 차로 표시
		return d.depth * 500;
	})
  .style("opacity", function(d, i){	// 깊이에 따라 투명도를 설정
		return d.depth / 10;	// 자식이 깊어지면 짙어짐
	})
// 맵 안에 문자 추가
tmap.enter()
  .append("text")	// text 요소를 추가
  .attr("class", "name")	// CSS 클래스 추가
  .attr("transform", function(d, i){	// 위치를 계산하여 X, Y 좌표를 일괄 설정
		var deg = 0;	// 각도는 0
		if (d.dx < d.dy){	// 만약 세로 방향 영역이라면 문자를 90도 회전
			deg = -90;
		}
		return "translate(" + (d.x+d.dx/2) + "," + (d.y+d.dy/2) + ") rotate("+deg+")";	// X, Y 좌표 설정
	})
  .attr("dy", "0.35em")	// 표시 위치 조정
  .text(function(d, i) {	// 문자 표시
		if ((d.depth == 0) || (d.children)) {	// 루트인가 자식 노드가 있는가
			return null;	// 루트이거나 자식 노드가 있다면 null을 반환하여 아무것도 표시하지 않음 
		}
		return d.name;	// 영역 안에 표시할 문자를 반환
	})
  .style("opacity", 0)	// 문자를 투명에서 불투명으로 함
  .transition()
  .delay(function(d, i) {	// 시간 차로 표시
		return d.depth * 500;
	})
  .style("opacity", 1.0)
