#![deny(clippy::all)]

use std::cell::RefCell;
use std::f64;
use std::rc::Rc;
use std::{thread, time};

use js_sys::Date;
use wasm_bindgen::__rt::core::cmp::Ordering::Equal;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::console;
use web_sys::{
    HtmlCanvasElement, HtmlInputElement, WebGlProgram, WebGlRenderingContext, WebGlShader,
};

use crate::data::Coordinate;

mod data;
//use web_sys::Window;

const RETINA: f64 = 3.;

fn pixel(px: f64) -> f64 {
    px * RETINA
}

fn window() -> web_sys::Window {
    web_sys::window().expect("no global `window` exists")
}

fn document() -> web_sys::Document {
    window().document().expect("no global `document` exists")
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    window()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("should register `requestAnimationFrame` OK");
}

fn canvas() -> HtmlCanvasElement {
    document()
        .get_element_by_id("canvas")
        .expect("no `canvas`")
        .dyn_into::<HtmlCanvasElement>()
        .expect("not html canvas element")
}

fn lon_slider() -> HtmlInputElement {
    document()
        .get_element_by_id("lon")
        .expect("no lon slider")
        .dyn_into::<HtmlInputElement>()
        .expect("not html input element")
}

fn lat_slider() -> HtmlInputElement {
    document()
        .get_element_by_id("lat")
        .expect("no lat slider")
        .dyn_into::<HtmlInputElement>()
        .expect("not html input element")
}

fn context() -> WebGlRenderingContext {
    canvas()
        .get_context("webgl")
        .unwrap()
        .unwrap()
        .dyn_into::<WebGlRenderingContext>()
        .ok()
        .expect("not a webglrenderingcontext")
}

pub fn compile_shader(
    context: &WebGlRenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, String> {
    let shader = context
        .create_shader(shader_type)
        .ok_or_else(|| String::from("Unable to create shader object"))?;
    context.shader_source(&shader, source);
    context.compile_shader(&shader);

    if context
        .get_shader_parameter(&shader, WebGlRenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(context
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| String::from("Unknown error creating shader")))
    }
}

pub fn link_program(
    context: &WebGlRenderingContext,
    vert_shader: &WebGlShader,
    frag_shader: &WebGlShader,
) -> Result<WebGlProgram, String> {
    let program = context
        .create_program()
        .ok_or_else(|| String::from("Unable to create shader object"))?;

    context.attach_shader(&program, vert_shader);
    context.attach_shader(&program, frag_shader);
    context.link_program(&program);

    if context
        .get_program_parameter(&program, WebGlRenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(context
            .get_program_info_log(&program)
            .unwrap_or_else(|| String::from("Unknown error creating program object")))
    }
}

fn create_lon_axis_program(context: &WebGlRenderingContext) -> Result<WebGlProgram, String> {
    let vert_shader = compile_shader(
        &context,
        WebGlRenderingContext::VERTEX_SHADER,
        r#"
        attribute vec4 aVertexPosition;
        uniform float uAspect;
        uniform float uLatTurn;
        uniform float uLonTurn;
        varying vec2 vCoord;
        varying float vRadius;
        varying float vDepth;

        void main() {
            float x = aVertexPosition.x;
            float y = aVertexPosition.y;
            float z = aVertexPosition.z;
            vCoord = vec2(x, z);
            vRadius = sqrt(1. - pow(y, 2.));
            float scale = uAspect < 1. ? 1. / uAspect : 1.;
            float a = radians(uLonTurn);
            gl_Position = vec4(x / uAspect, y, z, scale) *
            mat4(1,       0,        0,     0,
                 0,  cos(a),  -sin(a),     0,
                 0,  sin(a),   cos(a),     0,
                 0,       0,        0,     1);
            vDepth = gl_Position.z;
        }
    "#,
    )?;
    let frag_shader = compile_shader(
        &context,
        WebGlRenderingContext::FRAGMENT_SHADER,
        r#"
        const highp float cStroke = 0.005;
        varying highp vec2 vCoord;
        varying highp float vRadius;
        varying highp float vDepth;
        void main() {
            highp float maxRadius = vRadius;
            highp float minRadius = (1. - cStroke) * vRadius;
            highp float dist = sqrt(dot(vCoord,vCoord));
            if (dist > maxRadius || dist < minRadius) {
                discard;
            }
            highp float range = maxRadius - minRadius;
            highp float alpha = pow(1. - (abs(vRadius - dist) / range), 2.);
            highp float depthFactor = max(0.2, vDepth < 0. ? 1. : 1. - vDepth);
            gl_FragColor = vec4(0.3, 0.3, 0.3, alpha) * depthFactor;
        }
    "#,
    )?;
    link_program(context, &vert_shader, &frag_shader)
}

fn create_axis() -> impl Fn(f32, f32, f32) -> () {
    let context = context();
    let index_formula = [0, 1, 2, 0, 2, 3];
    let step = 10;
    let lon_quad_vertices: Vec<f32> = (step..180)
        .step_by(step)
        .enumerate()
        .flat_map(|(idx, deg)| {
            let lon = deg as i32 - 90;
            let y = -lon as f32 / 90.;
            vec![
                -1., y, -1., // front left
                -1., y, 1., // back left
                1., y, 1., // back right
                1., y, -1., // front right
            ]
        })
        .collect();
    let lon_quad_indices: Vec<u16> = (step..180)
        .step_by(step)
        .enumerate()
        .flat_map(|(idx, deg)| {
            let offset = idx * 4;
            vec![
                (index_formula[0] + offset) as u16,
                (index_formula[1] + offset) as u16,
                (index_formula[2] + offset) as u16,
                (index_formula[3] + offset) as u16,
                (index_formula[4] + offset) as u16,
                (index_formula[5] + offset) as u16,
            ]
        })
        .collect();

    let lon_axis_index_buffer = context.create_buffer();
    context.bind_buffer(
        WebGlRenderingContext::ELEMENT_ARRAY_BUFFER,
        lon_axis_index_buffer.as_ref(),
    );
    unsafe {
        let lon_index_array = js_sys::Uint16Array::view(&lon_quad_indices);
        context.buffer_data_with_array_buffer_view(
            WebGlRenderingContext::ELEMENT_ARRAY_BUFFER,
            &lon_index_array,
            WebGlRenderingContext::STATIC_DRAW,
        );
    }

    let lon_axis_vertex_buffer = context.create_buffer();
    context.bind_buffer(
        WebGlRenderingContext::ARRAY_BUFFER,
        lon_axis_vertex_buffer.as_ref(),
    );
    unsafe {
        let lon_vertex_array = js_sys::Float32Array::view(&lon_quad_vertices);
        context.buffer_data_with_array_buffer_view(
            WebGlRenderingContext::ARRAY_BUFFER,
            &lon_vertex_array,
            WebGlRenderingContext::STATIC_DRAW,
        );
    }

    let lon_axis_program = create_lon_axis_program(&context).unwrap();

    let aspect_location = context.get_uniform_location(&lon_axis_program, "uAspect");
    let lon_turn_location = context.get_uniform_location(&lon_axis_program, "uLonTurn");
    let lat_turn_location = context.get_uniform_location(&lon_axis_program, "uLatTurn");

    let lon_position_location =
        context.get_attrib_location(&lon_axis_program, "aVertexPosition") as u32;

    let n_vertices: i32 = lon_quad_indices.len() as i32;
    move |aspect, lat_turn, lon_turn| {
        context.use_program(Some(&lon_axis_program));
        context.uniform1f(aspect_location.as_ref(), aspect);
        context.uniform1f(lat_turn_location.as_ref(), lat_turn);
        context.uniform1f(lon_turn_location.as_ref(), lon_turn);
        context.bind_buffer(
            WebGlRenderingContext::ELEMENT_ARRAY_BUFFER,
            lon_axis_index_buffer.as_ref(),
        );
        context.bind_buffer(
            WebGlRenderingContext::ARRAY_BUFFER,
            lon_axis_vertex_buffer.as_ref(),
        );
        context.vertex_attrib_pointer_with_i32(
            lon_position_location,
            3,
            WebGlRenderingContext::FLOAT,
            false,
            0,
            0,
        );
        context.enable_vertex_attrib_array(lon_position_location);
        context.draw_elements_with_i32(
            WebGlRenderingContext::TRIANGLES,
            n_vertices,
            WebGlRenderingContext::UNSIGNED_SHORT,
            0,
        );
    }
}

fn create_routes_program(context: &WebGlRenderingContext) -> Result<WebGlProgram, String> {
    let vert_shader = compile_shader(
        &context,
        WebGlRenderingContext::VERTEX_SHADER,
        r#"
        attribute vec4 aVertexPosition;
        uniform float uAspect;
        uniform float uLatTurn;
        uniform float uLonTurn;
        varying float vDepth;

        void main() {
            float lat = aVertexPosition.x;
            float type = aVertexPosition.z;
            float thickness = (type < 2. || type > 4.) ? 0.04 : -0.04;
            float lon = aVertexPosition.y + thickness;
            float alpha = radians(lat);
            float beta = radians(lon + uLonTurn);
            float y = sin(alpha);
            float x = cos(alpha) * cos(beta);
            // todo: 0.0001 prevent blinking out of screen
            float z = cos(alpha) * sin(beta) + 0.0001;
            float a = radians(uLatTurn);
            float scale = uAspect < 1. ? 1. / uAspect : 1.;


            gl_Position = vec4(x / uAspect, y, z, scale) *
                mat4(1,       0,        0,     0,
                     0,  cos(a),  -sin(a),     0,
                     0,  sin(a),   cos(a),     0,
                     0,       0,        0,     1);
            vDepth = 1. - (gl_Position.z + 1.) / 2. ;
        }
    "#,
    )?;
    let frag_shader = compile_shader(
        &context,
        WebGlRenderingContext::FRAGMENT_SHADER,
        r#"
        varying highp float vDepth;
        void main() {
            highp float color = 1. - vDepth / 5.;
            highp float alpha = 0.5 + 0.3 * vDepth;
            gl_FragColor = vec4(color, color/2., 1. - vDepth, alpha);
        }
    "#,
    )?;
    link_program(context, &vert_shader, &frag_shader)
}

fn create_routes() -> impl Fn(f32, f32, f32) -> () {
    let context = context();

    let routes_vertices: Vec<f32> = data::lowp_flat_routes_vec3().collect();

    let routes_vertex_buffer = context.create_buffer();
    context.bind_buffer(
        WebGlRenderingContext::ARRAY_BUFFER,
        routes_vertex_buffer.as_ref(),
    );
    unsafe {
        let routes_vertex_array = js_sys::Float32Array::view(&routes_vertices);
        context.buffer_data_with_array_buffer_view(
            WebGlRenderingContext::ARRAY_BUFFER,
            &routes_vertex_array,
            WebGlRenderingContext::STATIC_DRAW,
        );
    }

    let routes_program = create_routes_program(&context).unwrap();

    let aspect_location = context.get_uniform_location(&routes_program, "uAspect");
    let lon_turn_location = context.get_uniform_location(&routes_program, "uLonTurn");
    let lat_turn_location = context.get_uniform_location(&routes_program, "uLatTurn");

    let routes_position_location =
        context.get_attrib_location(&routes_program, "aVertexPosition") as u32;

    let n_triangles: i32 = (routes_vertices.len() as f64 / 3.) as i32;
    move |aspect, lat_turn, lon_turn| {
        context.use_program(Some(&routes_program));
        context.uniform1f(aspect_location.as_ref(), aspect);
        context.uniform1f(lat_turn_location.as_ref(), lat_turn);
        context.uniform1f(lon_turn_location.as_ref(), lon_turn);
        context.bind_buffer(
            WebGlRenderingContext::ARRAY_BUFFER,
            routes_vertex_buffer.as_ref(),
        );
        context.vertex_attrib_pointer_with_i32(
            routes_position_location,
            3,
            WebGlRenderingContext::FLOAT,
            false,
            0,
            0,
        );
        context.enable_vertex_attrib_array(routes_position_location);
        context.draw_arrays(WebGlRenderingContext::TRIANGLES, 0, n_triangles);
    }
}

fn create_population_dot_program(context: &WebGlRenderingContext) -> Result<WebGlProgram, String> {
    let vert_shader = compile_shader(
        &context,
        WebGlRenderingContext::VERTEX_SHADER,
        r#"
        const float MAX_POP = 13639.57;

        attribute vec4 aVertexPosition;

        uniform float uPointScale;
        uniform float uAspect;
        uniform float uLonTurn;
        uniform float uLatTurn;

        varying highp float vDepth;
        void main() {
            float lat = aVertexPosition.x;
            float lon = aVertexPosition.y;
            float alpha = radians(lat);
            float beta = radians(lon + uLonTurn);
            float y = sin(alpha);
            float x = cos(alpha) * cos(beta);
            // todo: 0.0001 prevent blinking out of screen
            float z = cos(alpha) * sin(beta) + 0.0001;
            float a = radians(uLatTurn);
            float scale = uAspect < 1. ? 1. / uAspect : 1.;

            gl_Position = vec4(x / uAspect, y, z, scale) *
            mat4(1,       0,        0,     0,
                 0,  cos(a),  -sin(a),     0,
                 0,  sin(a),   cos(a),     0,
                 0,       0,        0,     1);

            vDepth = 1. - (gl_Position.z + 1.) / 2. ;

            float population = aVertexPosition.z;
            gl_PointSize = uPointScale * (9. +  150. * sqrt(population / MAX_POP) * pow(vDepth, 0.33));
        }
    "#,
    )?;
    let frag_shader = compile_shader(
        &context,
        WebGlRenderingContext::FRAGMENT_SHADER,
        r#"
        varying highp float vDepth;
        void main() {
            highp float x = 2. * (gl_PointCoord.x - 0.5);
            highp float y = 2. * (gl_PointCoord.y - 0.5);
            highp float radius = sqrt(pow(x, 2.) + pow(y, 2.));
            if (radius > 1.) {
                discard;
            }
            highp float alpha = (1.-radius) * (0.2 + 0.8 * pow(vDepth, 0.33));
            gl_FragColor = vec4(0., 0., vDepth, alpha);
        }
    "#,
    )?;
    link_program(context, &vert_shader, &frag_shader)
}

fn create_population_dot() -> impl Fn(f32, f32, f32, f32) -> () {
    let context = context();
    let vertices: Vec<f32> = data::lowp_flat_vec3().collect();
    // let maxPop = vertices
    //     .iter()
    //     .max_by(|a, b| a.partial_cmp(b).unwrap_or(Equal))
    //     .unwrap();
    // console::log_1(&format!("{}", maxPop).into());

    let vertex_buffer = context.create_buffer();
    context.bind_buffer(WebGlRenderingContext::ARRAY_BUFFER, vertex_buffer.as_ref());
    // Note that `Float32Array::view` is somewhat dangerous (hence the
    // `unsafe`!). This is creating a raw view into our module's
    // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
    // (aka do a memory allocation in Rust) it'll cause the buffer to change,
    // causing the `Float32Array` to be invalid.
    //
    // As a result, after `Float32Array::view` we have to be very careful not to
    // do any memory allocations before it's dropped.
    unsafe {
        let vert_array = js_sys::Float32Array::view(&vertices);
        context.buffer_data_with_array_buffer_view(
            WebGlRenderingContext::ARRAY_BUFFER,
            &vert_array,
            WebGlRenderingContext::STATIC_DRAW,
        );
    }

    let dot_program = create_population_dot_program(&context).unwrap();

    let lat_turn_location = context.get_uniform_location(&dot_program, "uLatTurn");
    let lon_turn_location = context.get_uniform_location(&dot_program, "uLonTurn");
    let aspect_location = context.get_uniform_location(&dot_program, "uAspect");
    let point_scale_location = context.get_uniform_location(&dot_program, "uPointScale");

    let position_location = context.get_attrib_location(&dot_program, "aVertexPosition") as u32;

    let n_vertices: i32 = (vertices.len() / 3) as i32;
    move |aspect, lat_turn, lon_turn, point_scale| {
        context.use_program(Some(&dot_program));
        context.uniform1f(aspect_location.as_ref(), aspect);
        context.uniform1f(lat_turn_location.as_ref(), lat_turn);
        context.uniform1f(lon_turn_location.as_ref(), lon_turn);
        context.uniform1f(point_scale_location.as_ref(), point_scale);
        context.bind_buffer(WebGlRenderingContext::ARRAY_BUFFER, vertex_buffer.as_ref());
        context.vertex_attrib_pointer_with_i32(
            position_location,
            3,
            WebGlRenderingContext::FLOAT,
            false,
            0,
            0,
        );
        context.enable_vertex_attrib_array(position_location);
        context.draw_arrays(WebGlRenderingContext::POINTS, 0, n_vertices);
    }
}

fn init() {
    let window_width = window()
        .inner_width()
        .ok()
        .expect("no inner width")
        .as_f64()
        .expect("not f64") as u32;
    let window_height = window()
        .inner_height()
        .ok()
        .expect("no inner height")
        .as_f64()
        .expect("not f64") as u32;
    canvas().set_width(pixel(window_width.into()) as u32);
    canvas().set_height(pixel(window_height.into()) as u32);

    let context = context();
    context.blend_func(
        WebGlRenderingContext::ONE,
        WebGlRenderingContext::ONE_MINUS_SRC_ALPHA,
    );
    context.enable(WebGlRenderingContext::BLEND);
}

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    init();

    // let axis_update = create_axis();
    let routes_update = create_routes();
    let population_dot_update = create_population_dot();

    let mut exit = false;
    let mut turn = ((Date::now() / 1000.) % 360.) as f32;
    let f = Rc::new(RefCell::new(None as Option<Closure<dyn FnMut()>>));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        if exit {
            // Drop our handle to this closure so that it will get cleaned
            // up once we return.
            let _ = f.borrow_mut().take();
            return;
        }
        let width = pixel(
            window()
                .inner_width()
                .expect("has no width")
                .as_f64()
                .expect("is not f64")
                .into(),
        );
        let height = pixel(
            window()
                .inner_height()
                .expect("has no height")
                .as_f64()
                .expect("is not f64")
                .into(),
        );

        let aspect = (width / height) as f32;
        let point_scale = (width.min(height) / pixel(800.)) as f32;
        let lat_turn = lat_slider()
            .value()
            .parse::<f32>()
            .expect("not valid f32 lat value");
        let lon_turn = turn
            + lon_slider()
                .value()
                .parse::<f32>()
                .expect("not valid f32 lon value");

        canvas().set_width(width as u32);
        canvas().set_height(height as u32);
        let context = context();
        context.viewport(0, 0, width as i32, height as i32);
        context.clear(WebGlRenderingContext::COLOR_BUFFER_BIT);

        // axis_update(aspect, lat_turn, lon_turn);
        routes_update(aspect, lat_turn, lon_turn);
        population_dot_update(aspect, lat_turn, lon_turn, point_scale);

        turn = (turn + 0.05) % 360.;
        exit = true;
        request_animation_frame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}
