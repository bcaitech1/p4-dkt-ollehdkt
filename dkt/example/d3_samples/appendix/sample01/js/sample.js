// 포스 레이아웃의 데이터셋
var dataSet = {
	nodes : [	// 노드 리스트
			{ name : "Apple" },	// 0번
			{ name : "Google" },	// 1번
			{ name : "Amazon" },	// 2번
			{ name : "Microsoft" }	// 3번
	],
	links : [	// 노드와노드를 연결하는 선의 관계. 배열 요소의 순서를 ID 참조 번호로 이용함
			{ source : 0, target : 1 },
			{ source : 1, target : 2 },
			{ source : 2, target : 3 },
			{ source : 3, target : 0 }
	]
}
// 포스 레이아웃 설정
var force = d3.layout.force()
  .nodes(dataSet.nodes)	// 노드 설정
  .links(dataSet.links)	// 노드와 노드를 연결하는 링크 선을 지정
  .size([320, 320])	// 표시 영역의 크기를 지정
  .linkDistance(90)	// 거리 지정
  .linkStrength(5)	// 강도 지정
  .gravity(0.0001)	// 중력 지정
  .start()
// 노드와 노드를 연결하는 선을 그림
var link = d3.select("#myGraph")
  .selectAll("line")	// 선 생성
  .data(dataSet.links)	// links 배열을 데이터셋으로 설정
  .enter()
  .append("line")	// 선 추가
  .attr("class", "forceLine")	// 선의 CSS 클래스를 지정
// 노드를 나타내는 원(●) 그리기
var node = d3.select("#myGraph")
  .selectAll("circle")	// 원 생성
  .data(dataSet.nodes)	// nodes 배열을 데이터셋으로 지정
  .enter()
  .append("circle")	// circle 추가
  .attr("r", 10)	// 반지름 설정
  .call(force.drag)	// 노드를 드래그할 수 있도록 함
// 다시 그릴 때(tick 이벤트가 발생할 때)에 선을 그림
force.on("tick", function() {
	link
	  .attr("x1", function(d) { return d.source.x; })	// 소스와 타깃 요소 좌표를 지정
	  .attr("y1", function(d) { return d.source.y; })
	  .attr("x2", function(d) { return d.target.x; })
	  .attr("y2", function(d) { return d.target.y; })
	node
	  .attr("cx", function(d) { return d.x; })	// 노드의 좌표를 지정
	  .attr("cy", function(d) { return d.y; })
})
