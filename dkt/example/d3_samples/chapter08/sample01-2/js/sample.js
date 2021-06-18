var svgWidth = 320;	// SVG 요소의 넓이
var svgHeight = 240;	// SVG 요소의 높이
var dataSet = [50, 30, 12, 5, 3];	// 데이터셋. 비율을 나타냄
// 원 그래프의 좌표값을 계산하는 메서드
var pie = d3.layout.pie()	// 원 그래프 레이아웃
  .value(function(d, i){ return d; })	// 데이터셋의 데이터 반환
// 원 그래프의 외경, 내경 설정
var arc = d3.svg.arc().innerRadius(0).outerRadius(100);
// 원 그래프 그리기
var pieElements = d3.select("#myGraph")
  .selectAll("path")	// path 요소 지정
  .data(pie(dataSet))	// 데이터를 요소에 연결
// 데이터 추가
pieElements.enter()	// 데이터 수만큼 반복
  .append("path")	// 데이터의 수만큼 path 요소가 추가됨
  .attr("class", "pie")	// CSS 클래스 설정
  .attr("d", arc)	// 부채꼴 지정
  .attr("transform", "translate("+svgWidth/2+", "+svgHeight/2+")")    // 원 그래프의 중심으로 함
