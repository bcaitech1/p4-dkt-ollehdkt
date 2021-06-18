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
// 팩 레이아웃
var bubble = d3.layout.pack()
  .size([320, 320])	// 표시 크기 지정
d3.select("#myGraph")
  .selectAll("circle")	// circle요소를 추가
  .data(bubble.nodes(dataSet))	// 데이터셋을 요소에 설정
  .enter()
  .append("circle")	// 데이터의 수만큼 circle 요소가 추가됨
  .attr("r", function(d){	// 반지름 지정
		return d.r;
	})
  .attr("cx", function(d, i){	// 중심 X 좌표 지정
		return d.x;	
	})
  .attr("cy", function(d, i){	// 中중심 Y 좌표 지정
		return d.y;	
	})

