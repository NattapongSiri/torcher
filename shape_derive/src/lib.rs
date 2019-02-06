extern crate proc_macro;
extern crate proc_macro2;
extern crate quote;
extern crate syn;

use quote::quote;
use proc_macro::TokenStream;
use proc_macro_hack::proc_macro_hack;
use syn::{bracketed, Expr, ExprArray, ExprIndex, ExprLit, ExprUnary, Lit, parse, parse_quote, RangeLimits, Token, UnOp};
use syn::parse::{Parse, ParseStream};

#[proc_macro_hack]
pub fn shape(input: TokenStream) -> TokenStream {
    let arr_expr: ExprArray = match parse::<Expr>(input).expect("Fail to parse shape as array expr") {
        Expr::Array(arr) => arr,
        _ => panic!("Support only [NUMERIC, NUMERIC..NUMERIC] format")
    };

    let mut ranges = Vec::with_capacity(arr_expr.elems.len());

    for elem in arr_expr.elems {
        let as_range = match elem {
            // Expr::Range(range) => {
            //     if range.from.is_none() && range.to.is_none() {
            //         // This is Rust syntax that return full size slice
            //         // We shall keep this semantic with tensor
            //         quote! {
            //             None
            //         }
            //     } else {
            //         let from = match range.from {
            //             Some(f) => quote!{#f},
            //             None => quote!{0}
            //         };
            //         let to = match range.to {
            //             Some(t) => Some(t),
            //             None => panic!("Missing max bound. The end of range is required if start of range is specified")
            //         };

            //         let to = match range.limits {
            //             RangeLimits::HalfOpen(_) => {
            //                 quote! {#to}
            //             },
            //             RangeLimits::Closed(_) => {
            //                 quote! {(#to + 1)}
            //             }
            //         };
            //         quote! {
            //             Some(#from..#to)
            //         }
            //     }
            // },
            Expr::Lit(ExprLit {lit: Lit::Int(int), ..}) => {
                
                quote! {
                    Some(#int)
                }
            },
            Expr::Unary(ExprUnary {op: UnOp::Neg(_), expr, ..}) => {
                match *expr {
                    Expr::Lit(ExprLit {lit: Lit::Int(_), ..}) => {
                        // it's some negative number.
                        // In PyTorch negative size in shape mean
                        // take all the rest remaining element
                        quote! {
                            None
                        }
                    },
                    _ => {
                        // it might be a positive if expression return negative num
                        quote! {
                            Some(#expr..(#expr + 1))
                        }
                    }
                }
            }
            _ => {
                quote! {
                    Some(#elem)
                }
            }
        };
        ranges.push(as_range);
    }

    let proper_shape = quote! {
        &[#(#ranges),*]
    };

    proper_shape.into()
}