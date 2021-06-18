// 누적 막대그래프의 데이터셋
var dataSet = [
    [{ year : 2010, p : 18 }, { year : 2011, p : 22 }, { year : 2012, p : 30 }, { year : 2013, p : 40 }],
    [{ year : 2010, p : 12 }, { year : 2011, p : 25 }, { year : 2012, p : 45 }, { year : 2013, p : 80 }],
    [{ year : 2010, p : 10 }, { year : 2011, p : 15 }, { year : 2012, p : 20 }, { year : 2013, p : 25 }]
];
// 그래프 관계의 데이터를 변수에 설정
var svgHeight = 240;	// svg 요소의 높이
var barWidth = 50;	// 막대그래프 가로 넓이
var step = 80;	// 막대그래프의 가로 간격
var offsetX = 10;	// X 좌표의 오프셋
// 색 설정
var color = ["red", "pink", "orange"];
// 누적 막대그래프의 레이아웃 설정
var stack = d3.layout.stack()   // 누적 막대그래프
  .y(function(d){
		return d.p;	// 키 이름 p의 데이터를 사용
	})
// 누적 막대그래프는 그룹으로 한꺼번에 표시
d3.select("#myGraph")
  .selectAll("g")	// 그래프 생성
  .data(stack(dataSet))	// 데이터셋 설정
  .enter()
  .append("g")	// g 요소 추가
  .attr("fill", function(d, i){   // 막대 그래프의 색 설정
        return color[i];
    })
  .selectAll("rect")
  .data(function(d, i){
		return d;	// 데이터를 하나 읽어옴(예：{ year : 2010, p : 18 })
	})
  .enter()
  .append("rect")	// 사각형 추가
  .attr("x", function(d, i){
		return offsetX + i * step;	// 막대그래프의 X 좌표를 반환
	})
  .attr("y", function(d, i){
		return svgHeight - d.y0 - d.y;	// 누적된 Y 좌표를 반환
	})
  .attr("width", barWidth)	// 막대그래프의 가로 넓이 설정
  .attr("height", function(d, i){
		return d.y;	// 막대그래프의 높이 반환
	})
