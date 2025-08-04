use gtether::render::uniform::{Uniform, UniformValue};
use gtether::render::{RenderTarget, Renderer, RendererStaleEvent};
use parry3d::na::{Isometry3, Point3, Translation3};
use std::sync::Arc;
use std::time::Duration;
use gtether::event::Event;
use vulkano::buffer::{AllocateBufferError, BufferContents};
use vulkano::Validated;

#[derive(BufferContents)]
#[repr(C)]
pub struct VP {
    pub matrix: glm::TMat4<f32>
}

#[derive(Debug, Clone)]
pub struct View {
    current_isometry: Isometry3<f32>,
    target_eye: Point3<f32>,
    target_target: Point3<f32>,
    target_isometry: Isometry3<f32>,
}

impl View {
    pub const UP: glm::TVec3<f32> = glm::TVec3::<f32>::new(0.0, 0.0, 1.0);

    pub fn new(
        eye: impl Into<Point3<f32>>,
        target: impl Into<Point3<f32>>,
    ) -> Self {
        let target_eye = eye.into();
        let target_target = target.into();
        let target_isometry = Isometry3::look_at_rh(
            &target_eye,
            &target_target,
            &Self::UP,
        );
        let current_isometry = target_isometry;
        Self {
            current_isometry,
            target_eye,
            target_target,
            target_isometry,
        }
    }

    #[inline]
    pub fn eye(&self) -> &Point3<f32> {
        &self.target_eye
    }

    #[inline]
    pub fn target(&self) -> &Point3<f32> {
        &self.target_target
    }

    #[inline]
    pub fn to_matrix(&self) -> glm::TMat4<f32> {
        self.current_isometry.to_matrix()
    }

    fn calc_target_isometry(&mut self) {
        self.target_isometry = Isometry3::look_at_rh(
            &self.target_eye,
            &self.target_target,
            &Self::UP,
        );
    }

    #[inline]
    pub fn set_eye(&mut self, eye: impl Into<Point3<f32>>) {
        self.target_eye = eye.into();
        self.calc_target_isometry();
    }

    #[inline]
    pub fn set_target(&mut self, target: impl Into<Point3<f32>>) {
        self.target_target = target.into();
        self.calc_target_isometry();
    }

    #[inline]
    pub fn translate(&mut self, translation: impl Into<Translation3<f32>>) {
        let translation = translation.into();
        self.target_eye = translation.transform_point(&self.target_eye);
        self.target_target = translation.transform_point(&self.target_target);
        self.calc_target_isometry();
    }

    #[inline]
    pub fn orient(&self) -> glm::TVec3<f32> {
        self.target_isometry.rotation
            .inverse_transform_vector(&glm::vec3(0.0, 0.0, -1.0))
    }

    #[inline]
    pub fn right(&self) -> glm::TVec3<f32> {
        self.target_isometry.rotation
            .inverse_transform_vector(&glm::vec3(1.0, 0.0, 0.0))
    }

    pub fn tick(&mut self, delta: Duration) {
        let factor = delta.as_secs_f32() * 5.0;
        self.current_isometry = match self.current_isometry.try_lerp_slerp(&self.target_isometry, factor, 0.01) {
            Some(iso) => iso,
            None => {
                // If rotation is 180 degrees, slerp is not well-defined. This is rare, so just
                //  shortcut by immediately jumping to the desired rotation.
                let tr = self.current_isometry
                    .translation
                    .vector
                    .lerp(&self.target_isometry.translation.vector, factor);
                Isometry3::from_parts(tr.into(), self.target_isometry.rotation)
            }
        };
    }
}

#[derive(Debug)]
pub struct Camera {
    view: View,
    perspective: glm::Mat4,
}

impl Camera {
    fn create_perspective(aspect_ratio: f32) -> glm::Mat4 {
        glm::Mat4::new_perspective(aspect_ratio, glm::half_pi(), 10.0, 10000.0)
    }

    pub fn new(
        render_target: &Arc<dyn RenderTarget>,
        eye: impl Into<Point3<f32>>,
        target: impl Into<Point3<f32>>,
    ) -> Self {
        let extent = render_target.extent().cast::<f32>();
        let view = View::new(eye, target);
        let perspective = Self::create_perspective(extent.x / extent.y);

        Self {
            view,
            perspective,
        }
    }

    pub fn new_uniform(
        renderer: &Arc<Renderer>,
        eye: impl Into<Point3<f32>>,
        target: impl Into<Point3<f32>>,
    ) -> Result<Arc<Uniform<VP, Self>>, AllocateBufferError> {
        let camera = Arc::new(Uniform::new(
            renderer.device().clone(),
            renderer.frame_manager(),
            Self::new(
                renderer.target(),
                eye,
                target,
            )
        ).map_err(Validated::unwrap)?);
        {
            let camera = camera.clone();
            renderer.event_bus().register(
                move |event: &mut Event<RendererStaleEvent>| {
                    camera.write().update(event.target());
                }
            ).unwrap();
        }
        Ok(camera)
    }

    #[inline]
    pub fn view(&self) -> &View {
        &self.view
    }

    #[inline]
    pub fn view_mut(&mut self) -> &mut View {
        &mut self.view
    }

    pub fn tick(&mut self, delta: Duration) {
        self.view.tick(delta);
    }

    fn update(&mut self, render_target: &Arc<dyn RenderTarget>) {
        let extent = render_target.extent().cast::<f32>();
        self.perspective = Self::create_perspective(extent.x / extent.y);
    }
}

impl UniformValue<VP> for Camera {
    #[inline]
    fn buffer_contents(&self) -> VP {
        VP {
            matrix: self.perspective * self.view.to_matrix(),
        }
    }
}