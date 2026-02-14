package org.bvd.appendix

import java.util.UUID

interface Aggregate<T> {

    fun getId(): UUID

    fun applyEvent(event: Event): T
}
