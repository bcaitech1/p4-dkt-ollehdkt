// 트리 레이아웃의 데이터셋
var dataSet = {
	name:"본사",
	children:[
		{ name:"경리부" },
		{ name:"업무부" },
		{ 
			name:"개발실",
			children:[
				{ name:"정보과" },
				{ name:"품질과" },
				{ name:"개발과",
					children:[
						{ name:"웹" },
						{ name:"애플리케이션" },
					]
				}
			]
		}
	]
}
// 그래프 관례 데이터를 변수에 설정
var svgWidth = 640;	// svg 요소의 넓이
var svgHeight = 480;	// svg 요소의 높이
var offsetLeft = svgWidth/2-100;	// 루트 노드의 왼쪽에서부터의 오프셋
var offsetTop = 50;	// 위로부터의 오프셋
var offsetBottom = 40;	// 아래로부터의 오프셋
var nSize = 25;	// ○의 크기
// 트리 레이아 지정
var tree = d3.layout.tree()
  .size([svgWidth, svgHeight-offsetTop - offsetBottom])	// 전체 크기 설정
  .nodeSize([120, 110])	// 노드 트리 전체 크기 설정
var nodes = tree.nodes(dataSet);	// 트리 노드 설정
// 노드 사이를 연결하는 선 표시
d3.select("#myGraph")
  .selectAll("path")	// path 요소를 대상으로 함
  .data(tree.links(nodes))	// 노드를 데이터셋으로 설정
  .enter()
  .append("path") // 패스 생성
  .attr("class", "line")	// 스타일시트 설정
  .attr("d", d3.svg.diagonal())//노드 사이를 연결
  .attr("transform", "translate(" + offsetLeft+", " + offsetTop+")") // 전체를 이동함
// 노드에 ○를 표시
d3.select("#myGraph")
  .selectAll("circle")	// circle 요소를 대상으로 함
  .data(nodes)	// 데이터셋을 설정
  .enter()
  .append("circle") // circle 요소 추가
  .attr("class", "node")	// 스타일시트 설정
  .attr("cx", function(d){ return d.x + offsetLeft })
  .attr("cy", function(d){ return d.y + offsetTop })
  .attr("r", nSize)
// 노드에 텍스트 표시
d3.select("#myGraph")
  .selectAll("text")	// text 요소를 대상으로 함
  .data(nodes)	// 데이터셋 설정
  .enter()
  .append("text") // text 요소 추가
  .attr("class", "label")	// 스타일시트 설정
  .attr("x", function(d){ return d.x + offsetLeft })
  .attr("y", function(d){ return d.y + offsetTop })
  .attr("dy", 6)	// 문자 위치 조정
  .text(function(d){ return d.name }) // name 속성 반환
