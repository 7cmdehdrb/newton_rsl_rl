# Newton Physics Simulation - 실행 및 구동 구조 가이드

이 문서는 `newton/newton/examples` 폴더에 위치한 예제 코드들의 구조를 분석하여, 사용자가 직접 원하는 시뮬레이션 환경과 시나리오를 구성할 수 있도록 돕는 실전 가이드입니다. 

## 1. 코드 구동 핵심 흐름
Newton 시뮬레이션 코드는 다음과 같은 공통 생명주기(Lifecycle)를 따릅니다.
1. **초기 설정 (Init)**: Warp 환경 설정, 시뮬레이션 파라미터(dt, fps 등) 정의
2. **모델 구성 (ModelBuilder)**: 월드 환경 생성, 물체(Body) 및 충돌 형상(Shape) 추가, 물리 속성 세팅
3. **솔버 및 상태 초기화**: VBD/XPBD 등 솔버 선택, 시뮬레이션 상태(`state_0`, `state_1`), 제어(`control`), 충돌 정보(`contacts`) 메모리 할당
4. **시뮬레이션 루프 (Run Loop)**: 매 프레임마다 여러 번의 서브스텝(Substeps) 물리 연산을 진행하고 뷰어(Viewer)를 통해 렌더링

---

## 2. 필수 모듈 임포트
최소한의 물리 시뮬레이션을 돌리기 위해서는 아래 패키지가 필요합니다.
(`newton.examples`는 예제 코드들에서 공통 기능을 묶어둔 헬퍼 모듈로, 실제 필수 의존성은 아닙니다. 전체적인 뼈대를 잡는 참고용으로 보시기 바랍니다.)
```python
import warp as wp
import newton
```

---

## 3. 커스텀 시나리오 작성 구조
예제 폴더의 파일들은 주로 `Example` 클래스 안에 데이터를 담아 구동하는 형태를 취하고 있습니다. **반드시 패키지(모듈) 구조나 이 클래스 형태를 따를 필요는 없으며**, 초기화와 루프의 흐름만 파악하여 자유롭게 스크립트를 작성하시면 됩니다. 아래는 예제 코드들이 공통적으로 취하고 있는 모범적인 구조(Best Practice)입니다.

### 3.1. 초기화 및 모델 생성 (`__init__` 혹은 스크립트 상단)
가장 먼저 호출되는 생성자로, 시뮬레이션 시간과 물체 배치를 수행합니다.
```python
class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_type = args.solver if hasattr(args, "solver") else "xpbd"
        
        # 1. 시뮬레이션 시간 설정
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # 2. 모델 빌더 생성
        builder = newton.ModelBuilder()

        # [옵션] 기본 물리 파라미터 튜닝 (VBD 등에서 접촉 안정성 향상을 위해 사용)
        builder.default_shape_cfg.ke = 1.0e6  # Contact stiffness
        builder.default_shape_cfg.kd = 1.0e1  # Contact damping
        builder.default_shape_cfg.mu = 0.5    # Friction coefficient

        # 3. 환경 및 물체 추가
        builder.add_ground_plane() # 바닥 평면 추가

        # 물체(Body) 생성 및 Shape 추가 (예시: 박스 생성)
        box_pos = wp.vec3(0.0, 0.0, 2.0)
        body_box = builder.add_body(
            xform=wp.transform(p=box_pos, q=wp.quat_identity()), 
            label="my_box"
        )
        builder.add_shape_box(body_box, hx=0.5, hy=0.5, hz=0.5)

        # 4. 모델 확정 (GPU 메모리 등 로딩)
        self.model = builder.finalize()

        # 5. 솔버 초기화 (XPBD 혹은 VBD)
        if self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(self.model, iterations=10)
        else:
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)

        # 6. 상태 변수 할당
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # 7. 뷰어 연동 및 카메라 세팅
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(5.0, -5.0, 2.0), pitch=-10.0, yaw=45.0)

        # GPU 연산 캡처 최적화 (Warp CUDA Graph)
        self.capture()
```

### 3.2. 물리 연산 스텝 (`simulate` 및 `step`)
시뮬레이션 로직은 한 프레임을 렌더링하기 전 내부적으로 여러 번 쪼개서(`sim_substeps`) 갱신합니다.
```python
    def simulate(self):
        # 여러 서브스텝에 걸쳐 물리엔진 동작
        for _ in range(self.sim_substeps):
            # 1. 외력 초기화
            self.state_0.clear_forces()
            
            # 뷰어 등 마우스 상호작용으로 가해지는 임의의 외력 적용
            self.viewer.apply_forces(self.state_0)
            
            # 2. 충돌 감지 파이프라인
            self.model.collide(self.state_0, self.contacts)
            
            # 3. 솔버 연산 수행 (과거상태, 다음상태, 제어입력, 충돌, 시간스텝)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            
            # 4. 상태 핑퐁 스왑 (계산이 완료된 1의 결과를 0로 넘김)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def capture(self):
        # CUDA 사용 중이라면 Graph 캡처를 지원하여 시뮬레이션 지연을 대폭 줄일 수 있습니다.
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def step(self):
        # 캡처된 그래프가 준비되어 있으면 즉시 실행, 없으면 일반 파이썬 루프 실행
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        
        # 내부 시간 업데이트
        self.sim_time += self.frame_dt
```

### 3.3. 시각화 갱신 (`render`)
뷰어에 현재 물체의 상태를 전달하여 화면에 모델을 그립니다.
```python
    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()
```

---

## 4. 메인 실행부
직접 스크립트를 돌릴 때는 아래와 같이 `newton.examples`의 헬퍼 루프(`run`)를 이용하거나, `while` 문을 이용해 직접 렌더링 루프를 작성할 수 있습니다.

**[옵션 A] 예제의 유틸리티(`newton.examples.run`)를 이용하는 경우:**
```python
import newton.examples

if __name__ == "__main__":
    # 1. 기본 인자 파서(Argument Parser) 생성
    parser = newton.examples.create_parser()
    
    # 2. Warp 디바이스 설정 및 뷰어 윈도우 생성
    viewer, args = newton.examples.init(parser)

    # 3. 작성한 시뮬레이션 클래스 초기화
    example = Example(viewer, args)

    # 4. Example을 관리하며 step()과 render()를 자동 반복하는 메인 루프 돌입
    newton.examples.run(example, args)
```

**[옵션 B] 직접 커스텀 루프를 돌리는 경우 (예제 의존성 제거):**
```python
if __name__ == "__main__":
    from newton.viewer import ViewerGL
    
    viewer = ViewerGL()
    # 자체 초기화 코드...
    
    while viewer.is_running():
        # 물리 스텝 실행
        # 시각화 렌더링
        viewer.begin_frame(sim_time)
        viewer.log_state(state)
        viewer.end_frame()
```

---

## 5. 자주 쓰는 모델링 레시피 (Shape 추가법)
`builder` 객체 내장 함수를 호출해 물체의 형태를 자유롭게 찍어낼 수 있습니다.
이 함수들은 **반드시 `builder.add_body` 직후**에 연결하여 물체 내부에 형상 데이터로서 종속시켜야 합니다.

- `builder.add_shape_box(body, hx, hy, hz)` : 직방체 (hx, hy, hz는 **절반** 길이 기준)
- `builder.add_shape_sphere(body, radius)` : 구
- `builder.add_shape_capsule(body, radius, half_height)` : 캡슐 모양
- `builder.add_shape_cylinder(body, radius, half_height)` : 원기둥
- `builder.add_shape_ellipsoid(body, a, b, c)` : 타원체
- `builder.add_shape_mesh(body, mesh)` : USD 파일 등에서 로드한 복잡한 폴리곤 메쉬

---

## 6. 스크립트 실행 방법
위 내용을 바탕으로 `my_scenario.py`를 작성했다면, 셸(CLI) 환경에서 쉽게 실행할 수 있습니다.

```bash
# 기본 OpenGL 뷰어로 인터랙티브하게 실행 (가장 일반적)
python my_scenario.py 

# VBD 솔버로 엔진을 교체하여 실행 (스크립트에 args.solver 분기가 있을 경우)
python my_scenario.py --solver vbd

# Headless 모드(화면 띄우지 않음) & USD 포맷 확장자로 프레임 굽기
python my_scenario.py --viewer usd --output-path result.usd
```
