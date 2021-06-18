var svgWidth = 320;	// SVG요소의 넓이
var svgHeight = 320;	// SVG요소의 높이
// 데이터셋
var dataSet = {
	value : 40,
	children : [
		{ value : 35 },
		{ value : 10 },
		{ value : 20 },
		{ value : 900,
			children : [
				{ value : 20 },
				{ value : 50 }
			]
		}
	]
}
// 색을 준비
var color = d3.scale.category10();  // 10색을 지정
// 팩 레이아웃
var bubble = d3.layout.pack()
  .size([320, 320])	// 표시 크기 지정
d3.select("#myGraph")
  .selectAll("circle")	// circle요소를 추가
  .data(bubble.nodes(dataSet))	// 데이터셋을 요소에 설정
  .enter()
  .append("circle")	// 데이터의 수만큼 circle 요소가 추가됨
  .attr("cx", function(d, i){	// 중심 X 좌표 지정
		return d.x;	
	})
  .attr("cy", function(d, i){	// 中중심 Y 좌표 지정
		return d.y;	
	})
  .attr("r", 0)	// 최초 반지름은 0으로 함
  .transition()
  .duration(function(d, i){	// 자식의 깊이에 따라 대기 시간을 설정
		return d.depth * 1000 + 500;	// 깊이×1초+0.5초
	})
  .attr("r", function(d){	// 반지름 지정
		return d.r;
	})
  .style("fill", function(d, i){
		return color(i);
	})

