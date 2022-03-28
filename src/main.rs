#[cfg(feature = "web")]
mod actix;
pub mod common;
#[cfg(feature = "consensus")]
mod consensus;
mod settings;
mod tonic;

#[cfg(feature = "consensus")]
use consensus::Consensus;
#[cfg(feature = "consensus")]
use slog::Drain;

use rand::{thread_rng, Rng};
use segment::fixtures::index_fixtures::{
    random_vector, FakeConditionChecker, TestRawScorerProducer,
};
use segment::index::hnsw_index::graph_layers::GraphLayers;
use segment::index::hnsw_index::point_scorer::FilteredScorer;
use segment::spaces::simple::{ EuclidMetric, CosineMetric, DotProductMetric };

use bit_vec::BitVec;
use segment::spaces::metric::Metric;
use segment::types::{Filter, PointOffsetType, VectorElementType};
use segment::vector_storage::simple_vector_storage::SimpleRawScorer;
use std::time::{Duration, Instant};

const M: usize = 4;
const EF_CONSTRUCT: usize = 200;
const USE_HEURISTIC: bool = true;

fn main() -> std::io::Result<()> {
    let start = Instant::now();

    let mut rng = thread_rng();

    let points: Vec<Vec<VectorElementType>> = vec![
        vec![0.851758, 0.909671],
        vec![0.823431, 0.372063],
        vec![0.97826, 0.933157],
        vec![0.39557, 0.306488],
        vec![0.230606, 0.634397],
        vec![0.514009, 0.399594],

        vec![0.354438, 0.762611],
        vec![0.0516154, 0.733427],
        vec![0.769864, 0.288072],
        vec![0.696896, 0.509403],
        vec![0.805918, 0.923242],
        vec![0.36507, 0.513271],
        vec![0.759294, 0.128909],
        vec![0.547961, 0.877969],
        vec![0.481519, 0.909654],
        vec![0.905498, 0.0285553],
        vec![0.452462, 0.346844],
        vec![0.826136, 0.84315],
        vec![0.350968, 0.92784],
        vec![0.309734, 0.531955],
        vec![0.2932, 0.186106],
        vec![0.754536, 0.228132],
        vec![0.151885, 0.107752],
        vec![0.0787534, 0.433617],
        vec![0.347133, 0.368639],
        
    ];

    let point_levels: Vec<usize> = vec![
        0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2,
    ];

    let mut graph_layers = GraphLayers::new(points.len(), M, M, EF_CONSTRUCT, 10, USE_HEURISTIC);
    let fake_condition_checker = FakeConditionChecker {};
    let deleted = BitVec::from_elem(points.len(), false);
    let metric = DotProductMetric {};
    for idx in 0..points.len() {
        let added_vector = points[idx].clone();
        let raw_scorer = SimpleRawScorer {
            query: metric.preprocess(&added_vector).unwrap_or(added_vector),
            metric: &metric,
            vectors: &points,
            deleted: &deleted,
        };
        let scorer = FilteredScorer::new(&raw_scorer, &fake_condition_checker, None);
        let level = graph_layers.get_random_layer(&mut rng);
        let level = point_levels[idx];
        // println!("Random level {} vs actual {}", rlevel, level);

        //println!("");
        //println!("Insert point {}", idx);
        graph_layers.link_new_point(idx as PointOffsetType, level, &scorer);
    }
    graph_layers.dump();

    let duration = start.elapsed();
    println!("Time elapsed is: {:?}", duration);

    Ok(())
}
