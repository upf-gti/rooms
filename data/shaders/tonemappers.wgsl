// Uncharted 2 tone map
// see: http://filmicworlds.com/blog/filmic-tonemapping-operators/
fn tonemap_uncharted2_imp( color : vec3f ) -> vec3f
{
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

fn tonemap_uncharted( c : vec3f ) -> vec3f
{
    let W = 11.2;
    let color = tonemap_uncharted2_imp(c * 2.0);
    let whiteScale = 1.0 / tonemap_uncharted2_imp( vec3f(W) );
    return color * whiteScale;//LINEARtoSRGB(color * whiteScale);
}

// Filmic Tonemapping Operators http://filmicworlds.com/blog/filmic-tonemapping-operators/
fn tonemap_filmic(x : vec3f) -> vec3f
{
  let X : vec3f = max(vec3f(0.0), x - 0.004);
  let result : vec3f = (X * (6.2 * X + 0.5)) / (X * (6.2 * X + 1.7) + 0.06);
  return pow(result, vec3f(2.2));
}